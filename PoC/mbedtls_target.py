import numpy as np
from binascii import hexlify
from lascar.tools.aes import sbox
import lascar
from rainbow import TraceConfig, Print, HammingWeight
from rainbow.devices import rainbow_stm32f215

print("Running trace generation for mbedtls implementation target...")

e = rainbow_stm32f215(print_config=Print.Functions, trace_config=TraceConfig(register=HammingWeight(), mem_value=HammingWeight()))
e.load("zephyr_mbedtls.elf")
e.setup()

key = b"\xde\xad\xbe\xef" * 4
print(key)

MBEDTLS_AES_FUNC = e.functions.get("mbedtls_aes_crypt_ecb")
if MBEDTLS_AES_FUNC is None:
    raise ValueError("'mbedtls_aes_crypt_ecb' not found in elf symbols!")
print(f"Cipher found at {hex(MBEDTLS_AES_FUNC)}")

def mbedtls_encrypt(key, plaintext):

    e.reset()

    # $: arm-none-eabi-nm build/zephyr/zephyr.elf | grep -C 5 key
    # $: 200001d4 D irk_key
    key_addr = 0x200001d4

    # mbedtls_aes_setkey_enc(mbedtls_aes_context *ctx, const unsigned char *key,
    #                        unsigned int keybits)
    ctx_addr = 0xDEAD0000
    e[ctx_addr] = 0

    AES_KEY_BITS = 128

    e['r0'] = ctx_addr
    e['r1'] = key_addr
    e['r2'] = AES_KEY_BITS
    e.start(e.functions['mbedtls_aes_setkey_enc'] | 1, 0)

    # int mbedtls_aes_crypt_ecb(mbedtls_aes_context *ctx,
    #                          int mode,
    #                          const unsigned char input[16],
    #                          unsigned char output[16])
    e['r0'] = ctx_addr    
    MBEDTLS_AES_ENCRYPT = 1
    e['r1'] = MBEDTLS_AES_ENCRYPT
    
    # Inject the generated plaintext
    pt_addr = 0xDEAD1000
    e[pt_addr] = plaintext

    e['r1'] = pt_addr

    e.start(e.functions['mbedtls_aes_crypt_ecb'] | 1, 0)
    
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

N = 5000

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

cont_lbls = [sbox[np.array(keys) ^ np.array(cont_lbls)]]

cont_trcs = [t for (t, _) in cont_traces]

cont_lbls=np.array(cont_lbls)
cont_trcs=np.array(cont_trcs)

print(f"CONTAINER_trc_shape: {cont_trcs.shape}")
print(f"CONTAINER_lbl_shape: {cont_lbls.shape}")

# Save the generated traces in the npz format
np.savez("container_traces_mbedtls", traces=cont_trcs, labels=cont_lbls)




from lascar import *

cpa_engines = [lascar.CpaEngine(name=f"CPA_{i}", 
                                selection_function=lambda plaintext, key_byte, index=i: sbox[plaintext[index] ^ key_byte],
                                guess_range=range(256)) for i in range(16)]

session = lascar.Session(CortexMAesContainer(N), engines=cpa_engines, name="lascar CPA").run()

guess_key = bytes([engine.finalize().max(1).argmax() for engine in cpa_engines])

print(f"Guessed key is : {hexlify(guess_key).upper()}")