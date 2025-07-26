import argparse
from datetime import datetime
from binascii import hexlify
from mbedtls_target_generator import MbedtlsTarget
from mbedtls_masked_target_generator import MbedtlsMaskedTarget
import numpy as np
import lascar
from tqdm import tqdm

from lascar.tools.aes import sbox

def generate_and_save_traces(target_object, container, num_traces, file_prefix, target, timestamp, chunk_size=5000):
    """
    Generates and saves a dataset of side-channel traces in chunks.

    Args:
        target_object: The SCA target object (profiling or attack target)
        container: The Lascar container object holding the value-leakage pairs
        num_traces: The total number of traces to generate
        file_prefix: A string for filename prefix (for example "profiling" or "attack")
        target: The target implementation string
        timestamp: The timestamp for unique filenames.
    """

    print(f"--- Generating {num_traces} {file_prefix} traces... ---")
    chunk_num = 0
    traces_all, labels_all, plaintext_all = [], [], []

    keys = profiling_target.key
    keys_ = np.array(list(keys), dtype=np.uint8)

    for idx, trace_obj in enumerate(tqdm(container)):
        plaintext = trace_obj.value
        plaintext_all.append(plaintext)
        traces_all.append(trace_obj.leakage)
        labels = sbox[keys_ ^ np.array(plaintext, dtype=np.uint8)]
        labels_all.append(labels)

        if len(traces_all) == chunk_size:
            print(f"Saving profiling chunk {chunk_num}...")
            np.savez(
                f"profiling_chunk_{chunk_num}_{target}_{timestamp}.npz",
                traces=traces_all,
                labels=labels_all,
                plaintexts=plaintext_all,
                key=profiling_target.key
            )
            traces_all = []
            labels_all = []
            plaintext_all = []
            chunk_num += 1
    
    if traces_all:
        print(f"Saving final chunk {chunk_num}...")
        np.savez(
                f"profiling_chunk_{chunk_num}_{target}_{timestamp}.npz",
                traces=traces_all,
                labels=labels_all,
                plaintexts=plaintext_all,
                key=profiling_target.key
            )
        
    print(f"--- Finished generating {file_prefix} traces. ---")
    

class CortexMAesContainer(lascar.AbstractContainer):
    """
    Lascar container for CortexM target.
    """
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
    parser.add_argument("--PLOT_LEAKAGE", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.target == "mbedtls":
        profiling_target = MbedtlsTarget(args.elf, args.N_PROFILING, verbose=False)
        attack_target = MbedtlsTarget(args.elf, args.N_ATTACK, verbose=False)
    elif args.target == "mbedtls_masked":
        profiling_target = MbedtlsMaskedTarget(args.elf, args.N_PROFILING, verbose=True)
        attack_target = MbedtlsMaskedTarget(args.elf, args.N_ATTACK, verbose=False)
    else:
        raise ValueError("Unsupported target.")

    print(f"Generating {args.N_PROFILING} profiling traces...")
    container = CortexMAesContainer(profiling_target, args.N_PROFILING)
    print(f"{args.N_PROFILING} traces generated")

    print("Labeling profiling traces...")
    k = [byte for byte in profiling_target.key]
    keys = [np.array(k)] * args.N_PROFILING
    
    generate_and_save_traces(profiling_target,
                             container,
                             args.N_PROFILING,
                             "profiling",
                             args.target,
                             timestamp)
    
    print(f"Generating {args.N_ATTACK} attack traces...")
    container = CortexMAesContainer(attack_target, args.N_ATTACK)
    print(f"{args.N_ATTACK} traces generated")

    print("Labeling attack traces...")
    k = [byte for byte in attack_target.key]
    keys = [np.array(k)] * args.N_ATTACK

    generate_and_save_traces(attack_target,
                             container,
                             args.N_ATTACK,
                             "attack",
                             args.target,
                             timestamp)

    if args.PLOT_LEAKAGE:
        profiling_target._plot_leakage(output_filename=f'annotated_trace_{args.target}_{timestamp}.png')

    if args.CPA_ATTACK:
        print(f"Attacking traces...")
        from lascar import *
        cpa_engines = [lascar.CpaEngine(name=f"CPA_{i}", 
                                        selection_function=lambda plaintext, key_byte, index=i: sbox[plaintext[index] ^ key_byte],
                                        guess_range=range(256)) for i in range(16)]

        session = lascar.Session(CortexMAesContainer(attack_target, args.N_ATTACK), engines=cpa_engines, name="lascar CPA").run()

        guess_key = bytes([engine.finalize().max(1).argmax() for engine in cpa_engines])

        print(f"Guessed key is : {hexlify(guess_key).upper()}")