

import lascar
import numpy as np
import numpy as np
from binascii import hexlify
from lascar.tools.aes import sbox
import lascar
import argparse
from rainbow import TraceConfig, Print, HammingWeight
from rainbow.devices import rainbow_stm32f215

from matplotlib import pyplot as plt

e = rainbow_stm32f215(print_config=Print.Functions, trace_config=TraceConfig(register=HammingWeight(), mem_value=HammingWeight()))
e.load("zephyr.elf")
e.setup()

def tinyaes_encrypt(key, plaintext):

    e.reset()

    #already hardcoded key in zephyr (see dissassembly)
    key_addr = 0x20000030

    # AES_init_ctx(struct AES_ctx* ctx, const uint8_t* key)
    ctx_addr = 0xDEAD0000
    e[ctx_addr] = 0

    e['r0'] = ctx_addr
    e['r1'] = key_addr
    e.start(e.functions['AES_init_ctx'] | 1, 0)

    #AES_ECB_encrypt(const struct AES_ctx* ctx, uint8_t* buf)
    e['r0'] = ctx_addr

    pt_addr = 0xDEAD1000
    e[pt_addr] = plaintext

    e['r1'] = pt_addr

    e.start(e.functions['AES_ECB_encrypt'] | 1, 0)
    
    trace = []
    for event in e.trace:
        if 'value' in event.keys():
            trace.append(event['value'])

    trace = np.array(trace) + np.random.normal(0, 1.0, (len(trace)))
    return trace

class CortexMAesContainer(lascar.AbstractContainer):

    def generate_trace(self, idx):
        plaintext = np.random.randint(0, 256, (16,), np.uint8)
        leakage = tinyaes_encrypt(key, plaintext.tobytes())
        return lascar.Trace(leakage, plaintext)

N = 1000
key = b"\xde\xad\xbe\xef" * 4


cpa_engines = [lascar.CpaEngine(name=f"CPA_{i}", selection_function=lambda plaintext, key_byte, index=i: sbox[plaintext[index] ^ key_byte], guess_range=range(256)) for i in range(16)]
s = lascar.Session(CortexMAesContainer(N), engines=cpa_engines, name="lascar CPA").run()

guess_key = bytes([engine.finalize().max(1).argmax() for engine in cpa_engines])
print("Guessed key is :", hexlify(guess_key).upper())

print("Correct key is :", hexlify(key).upper())