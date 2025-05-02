import numpy as np
from binascii import hexlify
from lascar.tools.aes import sbox
import lascar
from rainbow import TraceConfig, Print, HammingWeight
from rainbow.devices import rainbow_stm32f215

e = rainbow_stm32f215(print_config=Print.Functions, trace_config=TraceConfig(register=HammingWeight(), mem_value=HammingWeight()))
e.load("zephyr.elf")
e.setup()

key = b"\xde\xad\xbe\xef" * 4
print(key)

AES_CIPHER = e.functions.get("Cipher")
if AES_CIPHER is None:
    raise ValueError("'Cipher' not found in elf symbols!")
print(f"Cipher found at {hex(AES_CIPHER)}")

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


N = 5000

container = CortexMAesContainer(N)

print(f"CONTAINER_data: {container[24][0]}")
print(f"CONTAINER_meta: {container[24][1]}")

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

np.savez("container_traces", traces=cont_trcs, labels=cont_lbls)
