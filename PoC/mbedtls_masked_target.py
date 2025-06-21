import numpy as np
from binascii import hexlify
from lascar.tools.aes import sbox
import lascar
from rainbow import TraceConfig, Print, HammingWeight
from rainbow.devices import rainbow_stm32f215

print("Running trace generation for mbedtls implementation target...")

e = rainbow_stm32f215(print_config=Print.Functions, trace_config=TraceConfig(register=HammingWeight(), mem_value=HammingWeight()))
e.load("zephyr_mbedtls_masked.elf")
e.setup()

key = b"\xde\xad\xbe\xef" * 4
print(key)

MBEDTLS_AES_FUNC = e.functions.get("mbedtls_aes_crypt_ecb")
if MBEDTLS_AES_FUNC is None:
    raise ValueError("'mbedtls_aes_crypt_ecb' not found in elf symbols!")
print(f"mbedtls_aes_crypt_ecb found at {hex(MBEDTLS_AES_FUNC)}")

def mbedtls_encrypt(key, plaintext):

    e.reset()
    # e.emu.mem_map(0x20004000, 0x1000)

    # $: arm-none-eabi-nm build/zephyr/zephyr.elf | grep -C 5 key
    # $: 200001d4 D irk_key
    key_addr = 0x200001d4

    # key_addr = 0x20004000

    STACK_BASE = 0x20003000
    STACK_SIZE = 0x800
    e[STACK_BASE + STACK_SIZE] = 0


    # mbedtls_aes_setkey_enc(mbedtls_aes_context *ctx, const unsigned char *key,
    #                        unsigned int keybits)
    ctx_addr = 0xDEAD0000
    e[ctx_addr] = 0
    AES_CTX_SIZE = 277
    e[ctx_addr + AES_CTX_SIZE] = 0

    e[ctx_addr + e.PAGE_SIZE] = 0

    AES_KEY_BITS = 128

    e['r0'] = ctx_addr
    e['r1'] = key_addr
    e['r2'] = AES_KEY_BITS
    e.start(e.functions['mbedtls_aes_setkey_enc_masked'] | 1, 0)

    # int mbedtls_internal_aes_encrypt(mbedtls_aes_context *ctx,
    #                              const unsigned char input[16],
    #                              unsigned char output[16])
    e['r0'] = ctx_addr
    # Inject the generated plaintext
    pt_addr = 0xDEAD1000
    e[pt_addr] = plaintext
    e['r1'] = pt_addr

    # Avoid UC_ERR_WRITE_UNMAPPED error
    buf_out = 0xDEAD2000
    e[buf_out] = b"\x00" * 16
    e['r2'] = buf_out

    e.start(e.functions['mbedtls_internal_aes_encrypt_masked'] | 1, 0)
    
    trace = []
    for event in e.trace:
        if 'value' in event.keys():
            trace.append(event['value'])

    trace = np.array(trace) + np.random.normal(0, 1.0, (len(trace)))
    return trace

class CortexMAesContainer(lascar.AbstractContainer):

    def generate_trace(self, idx):
        plaintext = np.random.randint(0, 256, (16,), np.uint8)
        leakage = mbedtls_encrypt(key, plaintext.tobytes())
        return lascar.Trace(leakage, plaintext)

N = 1000

container = CortexMAesContainer(N)

print(f"CONTAINER_data: {container[24][0]}")
print(f"CONTAINER_meta: {container[24][1]}")

# Labelize the data with the SBOX intermediate value
cont_traces = [(t,l) for (t, l) in container]
cont_lbls = [np.array(l) for (_, l) in cont_traces]
k = [byte for byte in key]
keys = [np.array(k)] * len(cont_lbls)
print(f"keys - {len(keys)}")
print(f"cont_lbls - {len(cont_lbls)}")

plaintext = cont_lbls.copy()

# Apply the SBOX to the labels
cont_lbls = [sbox[np.array(keys) ^ np.array(cont_lbls)]]

cont_trcs = [t for (t, _) in cont_traces]

cont_lbls=np.array(cont_lbls)
cont_trcs=np.array(cont_trcs)

print(f"CONTAINER_trc_shape: {cont_trcs.shape}")
print(f"CONTAINER_lbl_shape: {cont_lbls.shape}")

# Save the generated traces in the npz format
np.savez("container_traces_mbedtls_masked_valid", traces=cont_trcs, 
                                                  labels=cont_lbls, 
                                                  keys=keys, 
                                                  plaintext=plaintext)




from lascar import *

cpa_engines = [lascar.CpaEngine(name=f"CPA_{i}", 
                                selection_function=lambda plaintext, key_byte, index=i: sbox[plaintext[index] ^ key_byte],
                                guess_range=range(256)) for i in range(16)]

session = lascar.Session(CortexMAesContainer(N), engines=cpa_engines, name="lascar CPA").run()

guess_key = bytes([engine.finalize().max(1).argmax() for engine in cpa_engines])

print(f"Guessed key is : {hexlify(guess_key).upper()}")
