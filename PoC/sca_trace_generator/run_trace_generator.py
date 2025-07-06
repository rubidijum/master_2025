import argparse
from datetime import datetime
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

    parser = argparse.ArgumentParser(description="Side-channel trace generator")
    parser.add_argument("--target", choices=["mbedtls", "mbedtls_masked"], required=True)
    parser.add_argument("--elf", required=True)
    parser.add_argument("--N_PROFILING", type=int, default=50000)
    parser.add_argument("--N_ATTACK", type=int, default=5000)
    parser.add_argument("--CPA_ATTACK", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.target == "mbedtls":
        profiling_target = MbedtlsTarget(args.elf, args.N_PROFILING, verbose=False)
        attack_target = MbedtlsTarget(args.elf, args.N_ATTACK, verbose=False)
    elif args.target == "mbedtls_masked":
        profiling_target = MbedtlsMaskedTarget(args.elf, args.N_PROFILING, verbose=False)
        attack_target = MbedtlsMaskedTarget(args.elf, args.N_ATTACK, verbose=False)
    else:
        raise ValueError("Unsupported target.")

    print(f"Generating {args.N_PROFILING} profiling traces...")
    container = CortexMAesContainer(profiling_target, args.N_PROFILING)
    print(f"{args.N_PROFILING} traces generated")

    print("Labeling profiling traces...")
    k = [byte for byte in profiling_target.key]
    keys = [np.array(k)] * args.N_PROFILING
    
    traces_all = []
    labels_all = []
    plaintext_all = []
    for trace_obj in container:
        plaintext_all.append(trace_obj.value)
        traces_all.append(trace_obj.leakage)
        labels_all.append([sbox[np.array(keys) ^ np.array(trace_obj.value)]])

    profiling_dataset_file = f"profiling_{args.target}_{timestamp}.npz"
    print(f"Saving profiling traces to {profiling_dataset_file}...")
    np.savez(
            profiling_dataset_file,
            traces=traces_all,
            labels=labels_all,
            plaintexts=plaintext_all,
            key=profiling_target.key
        )
    
    print(f"Generating {args.N_ATTACK} attack traces...")
    container = CortexMAesContainer(attack_target, args.N_ATTACK)
    print(f"{args.N_ATTACK} traces generated")

    print("Labeling attack traces...")
    k = [byte for byte in attack_target.key]
    keys = [np.array(k)] * args.N_ATTACK
    
    traces_all = []
    labels_all = []
    plaintext_all = []
    for trace_obj in container:
        plaintext_all.append(trace_obj.value)
        traces_all.append(trace_obj.leakage)
        labels_all.append([sbox[np.array(keys) ^ np.array(trace_obj.value)]])

    attack_dataset_file = f"attack_{args.target}_{timestamp}.npz"
    print(f"Saving attack traces to {attack_dataset_file}...")
    np.savez(
            attack_dataset_file,
            traces=traces_all,
            labels=labels_all,
            plaintexts=plaintext_all,
            key=attack_target.key
        )

    if args.CPA_ATTACK:
        print(f"Attacking traces...")
        from lascar import *
        cpa_engines = [lascar.CpaEngine(name=f"CPA_{i}", 
                                        selection_function=lambda plaintext, key_byte, index=i: sbox[plaintext[index] ^ key_byte],
                                        guess_range=range(256)) for i in range(16)]

        session = lascar.Session(CortexMAesContainer(attack_target, args.N_ATTACK), engines=cpa_engines, name="lascar CPA").run()

        guess_key = bytes([engine.finalize().max(1).argmax() for engine in cpa_engines])

        print(f"Guessed key is : {hexlify(guess_key).upper()}")