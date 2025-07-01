from binascii import hexlify
from mbedtls_target_generator import MbedtlsTarget
from mbedtls_masked_target_generator import MbedtlsMaskedTarget
import numpy as np
import lascar

from lascar.tools.aes import sbox

class CortexMAesContainer(lascar.AbstractContainer):
    def __init__(self, target_instance, num_traces):
        self.target = target_instance
        self.output_size = 16
        self.trace_count = num_traces
        super().__init__(num_traces)

    def generate_trace(self, idx):
        plaintext = np.random.randint(0, 256, (16,), np.uint8)
        self.target._encrypt_step(plaintext.tobytes())
        leakage = self.target._get_leakage()
        return lascar.Trace(leakage, plaintext)

if __name__ == "__main__":

    NUM_TRACES=1000

    print(f"Generating {NUM_TRACES} traces...")
    target = MbedtlsMaskedTarget("zephyr_mbedtls_masked.elf", key=b"\xAA"*16)

    container = CortexMAesContainer(target, NUM_TRACES)
    print(f"{NUM_TRACES} traces generated")

    print("Labeling traces...")
    k = [byte for byte in target.key]
    keys = [np.array(k)] * NUM_TRACES
    
    traces_all = []
    labels_all = []
    plaintext_all = []
    for trace_obj in container:
        plaintext_all.append(trace_obj.value)
        traces_all.append(trace_obj.leakage)
        labels_all.append([sbox[np.array(keys) ^ np.array(trace_obj.value)]])

    print(f"Saving traces...")
    np.savez(
            "test.npz",
            traces=traces_all,
            labels=labels_all,
            plaintexts=plaintext_all,
            key=target.key
        )


    print(f"Attacking traces...")
    from lascar import *
    cpa_engines = [lascar.CpaEngine(name=f"CPA_{i}", 
                                    selection_function=lambda plaintext, key_byte, index=i: sbox[plaintext[index] ^ key_byte],
                                    guess_range=range(256)) for i in range(16)]

    session = lascar.Session(CortexMAesContainer(target, NUM_TRACES), engines=cpa_engines, name="lascar CPA").run()

    guess_key = bytes([engine.finalize().max(1).argmax() for engine in cpa_engines])

    print(f"Guessed key is : {hexlify(guess_key).upper()}")