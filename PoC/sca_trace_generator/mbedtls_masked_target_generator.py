import numpy as np
from core_sca_generator import SideChannelTarget
import random as rand
import matplotlib.pyplot as plt
from rainbow import HammingWeight
import unicorn as uc

AES_KEY_BITS=128
class MbedtlsMaskedTarget(SideChannelTarget):

    @property
    def name(self):
        return "mbedtls"
    
    # def hook_instr(self, uc, address, size, user_data):
    #     if address in self.function_entry_addrs:
    #         fname = self.function_entry_addrs[address]
    #         if fname not in self.visited_functions:
    #             self.annotations.append((self.instruction_idx, fname))
    #             self.visited_functions.add(fname)
    #     self.instruction_idx += 1

    def _hook_function_entry(self, uc, address, size, user_data):
        # print(f"_hook_function_entry: accessing addr = {address}")
        # This hook triggers on every instruction. Check if the address
        # is a known function starting point.
        func_address_with_thumb_bit = address | 1

        if func_address_with_thumb_bit in self.function_entry_addrs:
            # print("Address is an entry point of function")
            fname = self.function_entry_addrs[func_address_with_thumb_bit]
                        
            # Record the sample index and the function name.
            self.pending_annotation_fname = self.function_entry_addrs[func_address_with_thumb_bit]

    def _hook_mem(self, uc, access, address, size, value, user_data):
        """
        Consumer Hook: This hook consumes pending annotations.
        """
        # leakage = HammingWeight()(value, size)
        # self.power_trace.append(leakage)
        
        # Next, check if a function call "flag" is raised.
        if self.pending_annotation_fname:
            # The correct sample index is the one we just added.
            power_sample_index = self.curr_trace_idx #len(self.power_trace) - 1

            # DEBUG print
            # print(f"Sample idx: {power_sample_index}")
            
            # Create the annotation with the correct timestamp.
            self.annotations.append((power_sample_index, self.pending_annotation_fname))
            
            # Lower the flag so we don't create duplicate annotations.
            self.pending_annotation_fname = None
        
        self.curr_trace_idx += 1
    
    def _load_target(self, target_path):
        self.emu.load(target_path)
        self.emu.setup()

        self.function_entry_addrs = {addr: name for name, addr in self.emu.functions.items()}

        # print(f"_load_target: Function entry addresses: {self.function_entry_addrs}")

        self.visited_functions = set()
        self.annotations = []
        self.instruction_idx = 0

        if "mbedtls_aes_setkey_enc_masked" not in self.emu.functions or \
            "mbedtls_internal_aes_encrypt_masked" not in self.emu.functions:
            raise ValueError("Required mbedtls functions not found in ELF symbols. Try recompiling and/or check symbol table.")
        
        self.key_mask = rand.randbytes(16)

        self.curr_trace_idx = 0

        self.emu.emu.hook_add(uc.UC_HOOK_CODE, self._hook_function_entry)
        self.emu.emu.hook_add(uc.UC_HOOK_MEM_READ | uc.UC_HOOK_MEM_WRITE, self._hook_mem)

    """DEBUG"""
    def _hook_for_watching_stack(self, uc, access, address, size, value, user_data):
        """This hook triggers ONLY when the watched stack address is written to."""
        pc = uc.reg_read(uc.UC_ARM_REG_PC)
        print(f"\n[!!!] STACK CORRUPTION DETECTED! [!!!]")
        print(f"      Instruction at 0x{pc:X} wrote to the watched stack address 0x{address:X}.")
        print(f"      This is the source of your bug. Find this address in your objdump.")
        # Stop the emulation to prevent the actual crash
        uc.emu_stop()

    def _hook_for_finding_stack(self, uc, address, size, user_data):
        """This hook runs only once to set up the watchpoint."""
        # Address of the push instruction at the start of the function
        target_function_entry = self.emu.functions['mbedtls_internal_aes_encrypt_masked'] | 1

        if address == target_function_entry:
            # We are at the entry of the function. The SP points to the top of our new stack frame.
            sp = uc.reg_read(uc.UC_ARM_REG_SP)
            print(f"[DEBUG] Entered target function at 0x{address:X}. Stack Pointer (SP) is at 0x{sp:X}.")
            print(f"[DEBUG] Setting a write watchpoint on the stack from 0x{sp:X} to 0x{sp+64:X}.")

            # Set the memory watchpoint on the stack frame.
            # We watch for any WRITES to the first 64 bytes of the stack.
            # The corrupted return address will be in here.
            self.write_hook = uc.hook_add(uc.UC_HOOK_MEM_WRITE, self._hook_for_watching_stack, begin=sp, end=sp + 64)

            # This hook has done its job, so remove it.
            uc.hook_del(self.setup_hook)
    """DEBUG"""  
    
    # TODO: Extract memory layout from this function
    # TODO: Add error checking for correct memory mappings (e.g. confirm that <disassembly_addr_of_x> is the same as <this_script_addr_of_x>)
    def _encrypt_step(self, plaintext: bytes):
        
        # Reset lists per encryption
        self.power_trace = []
        self.annotations = []
        self.pending_annotation_fname = None

        self.curr_trace_idx = 0

        self.emu.reset()

        # Define memory layout
        
        # Static addresses for mask injection
        # mask_key_addr = 0x00020d5c
        # mask_sbox_addr = 0x00020d6c
        # mask_subword_addr = 0x00020d7c
        # mask_r_addr = 0x00020d48

        unicorn_injected_mask_key_addr = 0x00020564 # D 
        unicorn_injected_mask_sbox_addr = 0x00020574 # D 
        unicorn_injected_mask_subword_addr = 0x00020584 # D 
        unicorn_r_addr = 0x00020550 # D 
        unicorn_r_in_addr = 0x00020561 # D 
        unicorn_r_out_addr = 0x00020560 # D 

        # AES algo adresses
        key_addr = 0x20000388
        ctx_addr = 0x20002000
        pt_addr = 0x20003000
        buf_out = 0x20004000

        # Avoid UC_ERR_WRITE_UNMAPPED error: trick Unicorn into mapping the page
        self.emu[ctx_addr] = b'\x00' * 560 #0

        # Avoid UC_ERR_WRITE_UNMAPPED error
        self.emu[buf_out] = b"\x00" * 16
        
        self.emu[key_addr] = self.key

        # Inject masks for key expansion
        self.emu[unicorn_injected_mask_key_addr] = self.key_mask
        
        # Inject masks for byte substitutions
        subword_mask = rand.randbytes(4)
        self.emu[unicorn_injected_mask_sbox_addr] = subword_mask

        sbox_mask = rand.randbytes(16)
        self.emu[unicorn_injected_mask_subword_addr] = sbox_mask

        # Inject initial state masks (r0 - r16)
        r_mask = rand.randbytes(16)
        self.emu[unicorn_r_addr] = r_mask

        # Inject r_in
        r_in = rand.randbytes(1)
        self.emu[unicorn_r_in_addr] = r_in

        # Inject r_out
        r_out = rand.randbytes(1)
        self.emu[unicorn_r_out_addr] = r_out

        # Setup the encryption key
        self.emu['r0'] = ctx_addr
        self.emu['r1'] = key_addr
        self.emu['r2'] = AES_KEY_BITS
        self.emu.start(self.emu.functions['mbedtls_aes_setkey_enc_masked'] | 1, 0)

        print("[DEBUG] Setting up stack watchpoint hooks...")
        self.setup_hook = self.emu.emu.hook_add(uc.UC_HOOK_CODE, self._hook_for_finding_stack)

        # Run encryption
        self.emu['r0'] = ctx_addr
        # Inject the generated plaintext
        self.emu[pt_addr] = plaintext
        self.emu['r1'] = pt_addr

        self.emu['r2'] = buf_out

        try:
            self.emu.start(self.emu.functions['mbedtls_internal_aes_encrypt_masked'] | 1, 0)
        finally:
            # Clean up the hooks after the run
            if hasattr(self, 'setup_hook'):
                self.emu.emu.hook_del(self.setup_hook)
            if hasattr(self, 'write_hook'):
                self.emu.emu.hook_del(self.write_hook)

        # Update the leakage
        for event in self.emu.trace:
            if 'value' in event:
                self.power_trace.append(event['value'])

    def _get_leakage(self):
        return np.array(self.power_trace) + np.random.normal(0, 1.0, (len(self.power_trace)))
        # trace = []
        # for event in self.emu.trace:
        #     if 'value' in event.keys():
        #         trace.append(event['value'])

        # trace = np.array(trace) + np.random.normal(0, 1.0, (len(trace)))
        # return trace

    def _plot_leakage(self):
        """Plots the trace and annotations collected by the hooks."""
        print("Plotting leakage with the producer-consumer hook method...")
        if not hasattr(self, 'power_trace') or not self.power_trace:
             self._encrypt_step(b'\x00' * 16)
        
        # The hooks have already built the lists with correct indices.
        trace_to_plot = np.array(self.power_trace)
        annotations_to_plot = self.annotations
        
        print(f"Plotting {len(trace_to_plot)} power samples and {len(annotations_to_plot)} annotations.")

        fig, ax = plt.subplots(figsize=(20, 6))
        ax.plot(trace_to_plot, label="Memory Read Trace", color="#a0c0c0", linewidth=0.7)

        for sample_index, fname in annotations_to_plot:
            if "mbedtls" in fname:
                fname = fname.split("mbedtls_")[-1]

            # Filter the non-relevant functions
            # TODO(avra): Move outside
            if "masked" not in fname and "mbedtls" not in fname:
                fname = ""

            ax.axvline(x=sample_index, color="lime", linestyle='--', linewidth=0.9, alpha=0.7)
            ax.text(sample_index, ax.get_ylim()[1], fname, rotation=60, verticalalignment='top',
                    color='black', fontsize=5)

        ax.set_title("Power Trace with Function Entry Points")
        ax.set_xlabel("Sample Index (Memory Read Events)")
        ax.set_ylabel("Value Read")
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
        fig.tight_layout()

        output_filename = "annotated_trace_final.png"
        fig.savefig(output_filename, dpi=200, bbox_inches='tight')
        print(f"Successfully saved new annotated trace to '{output_filename}'")