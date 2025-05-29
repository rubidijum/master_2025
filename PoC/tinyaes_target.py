class TinyAesTarget():
    """!
    Utility class for tinyaes implementation of AES algorithm
    """
    def __init__(self, emu):
        emulator = emu

    def tinyaes_encrypt(self, key, key_addr=0xDEAD0000, ctx_addr=0xDEAD1000, plaintext):

        self.emulator.reset()

        # Address of the secret key
        key_addr = 0xDEAD0000
        self.emulator[key_addr] = key

        # AES_init_ctx(struct AES_ctx* ctx, const uint8_t* key)
        ctx_addr = 0xDEAD1000
        self.emulator[ctx_addr] = 0

        self.emulator['r0'] = ctx_addr
        self.emulator['r1'] = key_addr
        e.start(e.functions['AES_init_ctx'] | 1, 0)

        #AES_ECB_encrypt(const struct AES_ctx* ctx, uint8_t* buf)
        self.emulator['r0'] = ctx_addr

        pt_addr = 0xDEAD2000
        self.emulator[pt_addr] = plaintext

        self.emulator['r1'] = pt_addr

        # Start execution at the AES_ECB_encrypt function
        # Thumb mode of execution - 1, 0
        self.emulator.start(e.functions['AES_ECB_encrypt'] | 1, 0)
        
        trace = []
        for event in e.trace:
            if 'value' in event.keys():
                trace.append(event['value'])

        trace = np.array(trace) + np.random.normal(0, 1.0, (len(trace)))
        return trace