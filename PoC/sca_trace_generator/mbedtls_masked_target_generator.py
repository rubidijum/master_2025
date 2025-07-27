import numpy as np
from core_sca_generator import SideChannelTarget
import random as rand
import matplotlib.pyplot as plt
from rainbow import HammingWeight
import unicorn as uc

from unicorn.arm_const import UC_ARM_REG_SP, UC_ARM_REG_LR, UC_ARM_REG_PC
from unicorn.unicorn_const import UC_HOOK_MEM_READ, UC_HOOK_MEM_WRITE

AES_KEY_BITS=128

MBEDTLS_AES_ENCRYPT=1
class MbedtlsMaskedTarget(SideChannelTarget):

    @property
    def name(self):
        return "mbedtls"

    def _hook_function_entry(self, uc, address, size, user_data):
        # This hook triggers on every instruction. Check if the address
        # is a known function starting point.
        func_address_with_thumb_bit = address | 1

        if func_address_with_thumb_bit in self.function_entry_addrs:
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
        self.current_masks = {}

        self.curr_trace_idx = 0

        # Annotate the functions
        self.emu.emu.hook_add(uc.UC_HOOK_CODE, self._hook_function_entry)
        self.emu.emu.hook_add(uc.UC_HOOK_MEM_READ | uc.UC_HOOK_MEM_WRITE, self._hook_mem)
    
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
        # NOTE: The memory layout changes based on the build environment, build target, target config, etc.
        # Make sure to check the expected values are at the correct addresses before running the emulation.
        EXIT_ADDRESS = 0xDEADBEEF


        # Optimizations off
        unicorn_injected_mask_key_addr = 0x0002ef7c # D 
        unicorn_injected_mask_sbox_addr = 0x0002ef8c # D 
        unicorn_injected_mask_subword_addr = 0x0002ef9c # D 
        unicorn_r_addr = 0x0002ef68 # D 
        unicorn_r_in_addr = 0x0002ef79 # D 
        unicorn_r_out_addr = 0x0002ef78 # D 

        # AES algo adresses
        key_addr = 0x20000188
        # 200023f8 B __bss_end
        end_of_bss = 0x200023f8
        safe_start_addr = (end_of_bss + 3) & ~0x3

        ctx_addr = safe_start_addr
        pt_addr = ctx_addr + 560
        buf_out = pt_addr + 16
        

        # Avoid UC_ERR_WRITE_UNMAPPED error: map enough for the mbedtls_aes_context struct
        # nr = 8 bits
        # rk_offset = 32 bits
        # buf_masked = 44 * 32 bits
        # buf_mask = 44 * 32 bits
        # buf = 44 * 32 bits
        # 533 bytes
        self.emu[ctx_addr] = b'\x00' * 560

        # Avoid UC_ERR_WRITE_UNMAPPED error
        self.emu[buf_out] = b"\x00" * 16
        
        self.current_masks = {
            'key_mask': rand.randbytes(16),
            'subword_mask': rand.randbytes(4),
            'sbox_mask': rand.randbytes(16),
            'r_mask': rand.randbytes(16),
            'r_in': rand.randbytes(1),
            'r_out': rand.randbytes(1)
        }
        
        self.emu[key_addr] = self.key

        # Inject masks for key expansion
        self.emu[unicorn_injected_mask_key_addr] = self.key_mask
        
        # Inject masks for word substitutions (within key expansion)
        self.emu[unicorn_injected_mask_subword_addr] = self.current_masks['subword_mask']

        # Inject SBOX masks
        self.emu[unicorn_injected_mask_sbox_addr] = self.current_masks['sbox_mask']

        # Inject initial state masks (r0 - r16)
        self.emu[unicorn_r_addr] = self.current_masks['r_mask']

        # Inject r_in
        self.emu[unicorn_r_in_addr] = self.current_masks['r_in']

        # Inject r_out
        self.emu[unicorn_r_out_addr] = self.current_masks['r_out']

        print("\n" + "="*60)
        print("--- Running Key Schedule ---")

        # Setup the encryption key
        self.emu['r0'] = ctx_addr
        self.emu['r1'] = key_addr
        self.emu['r2'] = AES_KEY_BITS
        self.emu.emu.reg_write(UC_ARM_REG_LR, EXIT_ADDRESS)
        self.emu.start(self.emu.functions['mbedtls_aes_setkey_enc'] | 1, 0)

        print("--- Key Schedule Complete ---")
        print("="*60 + "\n")

        # Focus only on the encryption
        self.emu.trace.clear()
        self.power_trace = []
        self.annotations = []
        self.curr_trace_idx = 0

        # Run encryption
        self.emu['r0'] = ctx_addr
        # int mode
        self.emu['r1'] = MBEDTLS_AES_ENCRYPT
        # Inject the generated plaintext
        self.emu[pt_addr] = plaintext
        self.emu['r2'] = pt_addr

        self.emu['r3'] = buf_out

        print("\n\n" + "*"*60)
        print("--- Running Encryption ---")

        try:
            self.emu.emu.reg_write(UC_ARM_REG_LR, EXIT_ADDRESS)
            self.emu.start(self.emu.functions['mbedtls_aes_crypt_ecb'] | 1, 0)
        finally:
            # Clean up the hooks after the run
            if hasattr(self, 'setup_hook'):
                self.emu.emu.hook_del(self.setup_hook)
            if hasattr(self, 'write_hook'):
                self.emu.emu.hook_del(self.write_hook)

        print("--- Encryption Complete ---")
        print("*"*60 + "\n\n")

        # Update the leakage
        for event in self.emu.trace:
            if 'value' in event:
                self.power_trace.append(event['value'])

    def _get_leakage(self):
        return np.array(self.power_trace) + np.random.normal(0, 1.0, (len(self.power_trace)))

    def _plot_leakage(self, output_filename = "annotated_trace_final.png", show_plt=False):
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
        ax.set_xlabel("Memory R/W Events")
        ax.set_ylabel("Memory Value")
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
        fig.tight_layout()

        if show_plt:
            fig.show()

        fig.savefig(output_filename, dpi=200, bbox_inches='tight')
        print(f"Successfully saved new annotated trace to '{output_filename}'")