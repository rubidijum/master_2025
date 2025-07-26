import numpy as np
from core_sca_generator import SideChannelTarget
import unicorn as uc
import matplotlib.pyplot as plt

from unicorn.arm_const import UC_ARM_REG_SP, UC_ARM_REG_LR, UC_ARM_REG_PC

AES_KEY_BITS=128

class MbedtlsTarget(SideChannelTarget):

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
        # Next, check if a function call "flag" is raised.
        if self.pending_annotation_fname:
            power_sample_index = self.curr_trace_idx
            
            # Create the annotation with the correct timestamp.
            self.annotations.append((power_sample_index, self.pending_annotation_fname))
            
            # Lower the flag so we don't create duplicate annotations.
            self.pending_annotation_fname = None
        
        self.curr_trace_idx += 1
    
    def _load_target(self, target_path):
        self.emu.load(target_path)
        self.emu.setup()

        self.function_entry_addrs = {addr: name for name, addr in self.emu.functions.items()}

        self.annotations = []
        self.instruction_idx = 0

        if "mbedtls_aes_setkey_enc" not in self.emu.functions or \
            "mbedtls_internal_aes_encrypt" not in self.emu.functions:
            raise ValueError("Required mbedtls functions not found in ELF symbols. Try recompiling and/or check symbol table.")
        
        self.curr_trace_idx = 0

        # Annotate the functions
        self.emu.emu.hook_add(uc.UC_HOOK_CODE, self._hook_function_entry)
        self.emu.emu.hook_add(uc.UC_HOOK_MEM_READ | uc.UC_HOOK_MEM_WRITE, self._hook_mem)
        
    def _encrypt_step(self, plaintext: bytes):
        
        # Reset lists per encryption
        self.power_trace = []
        self.annotations = []
        self.pending_annotation_fname = None

        self.curr_trace_idx = 0
        
        self.emu.reset()

        EXIT_ADDRESS = 0xDEADBEEF

        #TODO: Load symbols programmatically
        # Define memory layout
        key_addr = 0x20000388 #0x200001d4
        ctx_addr = 0xDEAD0000
        pt_addr = 0xDEAD1000
        buf_out = 0xDEAD2000

        # Avoid UC_ERR_WRITE_UNMAPPED error: trick Unicorn into mapping the page
        self.emu[ctx_addr] = 0
        # self.emu[ctx_addr + self.emu.PAGE_SIZE] = 0
        
        self.emu[key_addr] = self.key

        # Setup encryption key
        self.emu['r0'] = ctx_addr
        self.emu['r1'] = key_addr
        self.emu['r2'] = AES_KEY_BITS
        self.emu.emu.reg_write(UC_ARM_REG_LR, EXIT_ADDRESS)
        self.emu.start(self.emu.functions['mbedtls_aes_setkey_enc'] | 1, 0)

        # Run encryption
        self.emu['r0'] = ctx_addr
        self.emu[pt_addr] = plaintext
        self.emu['r1'] = pt_addr
        self.emu[buf_out] = b"\x00" * 16 # Avoid write errors
        self.emu['r2'] = buf_out

        self.emu.emu.reg_write(UC_ARM_REG_LR, EXIT_ADDRESS)
        self.emu.start(self.emu.functions['mbedtls_internal_aes_encrypt'] | 1, 0)

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

                ax.axvline(x=sample_index, color="lime", linestyle='--', linewidth=0.9, alpha=0.7)
                ax.text(sample_index, ax.get_ylim()[1], fname, rotation=60, verticalalignment='top',
                        color='black', fontsize=5)

            ax.set_title("Power Trace with Function Entry Points")
            ax.set_xlabel("Sample Index (Memory Read Events)")
            ax.set_ylabel("Value Read")
            ax.legend()
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
            fig.tight_layout()

            if show_plt:
                fig.show()

            fig.savefig(output_filename, dpi=200, bbox_inches='tight')
            print(f"Successfully saved new annotated trace to '{output_filename}'")