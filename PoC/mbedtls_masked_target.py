import numpy as np
from binascii import hexlify
from lascar.tools.aes import sbox
import lascar
from rainbow import TraceConfig, Print, HammingWeight
from rainbow.devices import rainbow_stm32f215

import argparse

# Argument parser
parser = argparse.ArgumentParser(description="Trace generation for AES SCA")

parser.add_argument("--CPA_ATTACK", action="store_true",
                    help="Run CPA attack on generated traces.")

parser.add_argument("--N_PROFILING", type=int, default=50000,
                    help="Number of profiling traces to generate.")

parser.add_argument("--N_ATTACK", type=int, default=5000,
                    help="Number of attack traces to generate.")

args = parser.parse_args()

# Safety check
if args.N_PROFILING < args.N_ATTACK:
    raise ValueError(f"N_PROFILING ({args.N_PROFILING}) must be >= N_ATTACK ({args.N_ATTACK})")

print(f"[CONFIG] Profiling: {args.N_PROFILING} | Attack: {args.N_ATTACK} | CPA: {args.CPA_ATTACK}")

print("Running trace generation for masked mbedtls implementation target...")

e = rainbow_stm32f215(print_config=Print.Functions, trace_config=TraceConfig(register=HammingWeight(), mem_value=HammingWeight()))
e.load("zephyr_mbedtls_masked.elf")
e.setup()

key = b"\xde\xad\xbe\xef" * 4
print(f"Fixed key: {key}")

MBEDTLS_AES_FUNC = e.functions.get("mbedtls_aes_crypt_ecb")
if MBEDTLS_AES_FUNC is None:
    raise ValueError("'mbedtls_aes_crypt_ecb' not found in elf symbols!")
print(f"mbedtls_aes_crypt_ecb found at {hex(MBEDTLS_AES_FUNC)}")

def mbedtls_encrypt(key, plaintext):

    e.reset()
    # e.emu.mem_map(0x20004000, 0x1000)

    # Since Unicorn doesn't emulate the entropy =>
    # Before running the encryption, inject the masks into the memory.
    mask_addr = 0x20004000
    mask = np.random.randint(0, 256, (16,), np.uint8)
    e[mask_addr] = mask.tobytes()

    # $: arm-none-eabi-nm build/zephyr/zephyr.elf | grep -C 5 key
    # $: 200001d4 D irk_key
    key_addr = 0x20000388

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

container_profiling = CortexMAesContainer(args.N_PROFILING)
container_attack = CortexMAesContainer(args.N_ATTACK)

def labelize_trace(container):
    """
    Labelize the trace with the SBOX intermediate value.
    """
    traces = [(t, l) for (t, l) in container]
    lbls = [np.array(l) for (_, l) in traces]
    k = [byte for byte in key]
    keys = [np.array(k)] * len(lbls)
    print(f"keys len - {len(keys)}")
    print(f"lbls len - {len(lbls)}")

    plaintext = lbls.copy()

    # Apply the SBOX to the labels
    lbls = [sbox[np.array(keys) ^ np.array(lbls)]]

    trcs = [t for (t, _) in traces]

    lbls = np.array(lbls)
    trcs = np.array(trcs)

    print(f"trc_shape: {trcs.shape}")
    print(f"lbl_shape: {lbls.shape}")

    return trcs, lbls, keys, plaintext

## Profiling traces

profiling_traces, profiling_labels, profiling_keys, profiling_plaintext = labelize_trace(container_profiling)
# Save the generated traces in the npz format
np.savez("profiling_traces_mbedtls_masked_valid", traces=profiling_traces,
                                                  labels=profiling_labels, 
                                                  keys=profiling_keys, 
                                                  plaintext=profiling_plaintext)

## Attack traces

attack_traces, attack_labels, attack_keys, attack_plaintext = labelize_trace(container_attack)
# Save the generated traces in the npz format
np.savez("attack_traces_mbedtls_masked_valid", traces=attack_traces,
                                                  labels=attack_labels, 
                                                  keys=attack_keys, 
                                                  plaintext=attack_plaintext)


if args.CPA_ATTACK:
    from lascar import *

    cpa_engines = [lascar.CpaEngine(name=f"CPA_{i}", 
                                    selection_function=lambda plaintext, key_byte, index=i: sbox[plaintext[index] ^ key_byte],
                                    guess_range=range(256)) for i in range(16)]

    session = lascar.Session(CortexMAesContainer(args.N_PROFILING), engines=cpa_engines, name="lascar CPA").run()

    guess_key = bytes([engine.finalize().max(1).argmax() for engine in cpa_engines])

    print(f"Guessed key is : {hexlify(guess_key).upper()}")
