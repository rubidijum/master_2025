import numpy as np
from core_sca_generator import SideChannelTarget

AES_KEY_BITS=128

class MbedtlsTarget(SideChannelTarget):

    @property
    def name(self):
        return "mbedtls"
    
    def _load_target(self, target_path):
        self.emu.load(target_path)
        self.emu.setup()

        if "mbedtls_aes_setkey_enc" not in self.emu.functions or \
            "mbedtls_internal_aes_encrypt" not in self.emu.functions:
            raise ValueError("Required mbedtls functions not found in ELF symbols. Try recompiling and/or check symbol table.")
        
    def _encrypt_step(self, plaintext: bytes):
        self.emu.reset()

        # Define memory layout
        key_addr = 0x200001d4
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
        self.emu.start(self.emu.functions['mbedtls_aes_setkey_enc'] | 1, 0)

        # Run encryption
        self.emu['r0'] = ctx_addr
        self.emu[pt_addr] = plaintext
        self.emu['r1'] = pt_addr
        self.emu[buf_out] = b"\x00" * 16 # Avoid write errors
        self.emu['r2'] = buf_out
        self.emu.start(self.emu.functions['mbedtls_internal_aes_encrypt'] | 1, 0)

    def _get_leakage(self):
        trace = []
        for event in self.emu.trace:
            if 'value' in event.keys():
                trace.append(event['value'])

        trace = np.array(trace) + np.random.normal(0, 1.0, (len(trace)))
        return trace