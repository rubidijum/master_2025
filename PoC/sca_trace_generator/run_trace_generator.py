import argparse
from datetime import datetime
from binascii import hexlify
from mbedtls_target_generator import MbedtlsTarget
from mbedtls_masked_target_generator import MbedtlsMaskedTarget
import numpy as np
import lascar
from tqdm import tqdm
import h5py
import os

from lascar.tools.aes import sbox

saved_annotations = None

def append_chunk_to_h5(h5_path, traces_chunk, plaintexts_chunk, labels_chunk, rin_chunk, rout_chunk):
    with h5py.File(h5_path, 'a') as hf:
        current_size = hf['traces'].shape[0]
        chunk_size = len(traces_chunk)

        hf['traces'].resize((current_size + chunk_size, hf['traces'].shape[1]))
        hf['plaintexts'].resize((current_size + chunk_size, hf['plaintexts'].shape[1]))
        hf['labels'].resize((current_size + chunk_size, hf['labels'].shape[1]))

        hf['traces'][current_size:] = np.array(traces_chunk)
        hf['plaintexts'][current_size:] = np.array(plaintexts_chunk)
        hf['labels'][current_size:] = np.array(labels_chunk)

        rin_as_integers = np.array([b[0] for b in rin_chunk], dtype=np.uint8)
        rout_as_integers = np.array([b[0] for b in rout_chunk], dtype=np.uint8)

        hf['masks_rin'].resize((current_size + chunk_size, 1))
        hf['masks_rout'].resize((current_size + chunk_size, 1))

        hf['masks_rin'][current_size:] = rin_as_integers.reshape(-1, 1)
        hf['masks_rout'][current_size:] = rout_as_integers.reshape(-1, 1)

def generate_h5_dataset(target_object, container, num_traces, output_filename, chunk_size=5000):

    print(f"--- Generating {num_traces} traces for HDF5 file: {output_filename} ---")

    first_trace_obj = next(iter(container))

    first_leakage = target_object._get_leakage()
    saved_annotations = target_object.annotations

    with h5py.File(output_filename, 'w') as hf:
        hf.create_dataset('traces', shape=(0,) + first_leakage.shape, maxshape=(None,) + first_leakage.shape, dtype=first_leakage.dtype)
        hf.create_dataset('plaintexts', shape=(0, 16), maxshape=(None, 16), dtype=np.uint8)
        hf.create_dataset('labels', shape=(0, 16), maxshape=(None, 16), dtype=np.uint8)
        
        # Create datasets for each per-trace mask
        hf.create_dataset('masks_rin', shape=(0, 1), maxshape=(None, 1), dtype=np.uint8)
        hf.create_dataset('masks_rout', shape=(0, 1), maxshape=(None, 1), dtype=np.uint8)

        hf.create_dataset('key', data=list(target_object.key))
        
        if saved_annotations:
            annotation_idcs = [item[0] for item in saved_annotations]
            annotation_names = [item[1] for item in saved_annotations]
        
            hf.create_dataset('annotations_idcs', data=np.array(annotation_idcs, dtype=np.uint32))
            hf.create_dataset('annotations_names', data=np.array(annotation_names, dtype=h5py.special_dtype(vlen=str)))
        else:
            print("Warning: No annotations generated.")

    keys_ = np.array(list(target_object.key), dtype=np.uint8)

    traces_chunk, labels_chunk, plaintexts_chunk = [], [], []
    rin_chunk, rout_chunk = [], []

    # Handle first chunk
    traces_chunk.append(first_leakage)
    plaintexts_chunk.append(first_trace_obj.value)
    labels_chunk.append(sbox[keys_ ^ np.array(first_trace_obj.value, dtype=np.uint8)])
    rin_chunk.append(target_object.current_masks['r_in'])
    rout_chunk.append(target_object.current_masks['r_out'])

    for trace_obj in tqdm(container, desc='Generating remaining traces', total=num_traces, initial=1):
        leakage = target_object._get_leakage()
        
        traces_chunk.append(leakage)
        plaintexts_chunk.append(trace_obj.value)
        labels_chunk.append(sbox[keys_ ^ np.array(trace_obj.value, dtype=np.uint8)])
        
        # Append the individual masks
        rin_chunk.append(target_object.current_masks['r_in'])
        rout_chunk.append(target_object.current_masks['r_out'])

        if len(traces_chunk) == chunk_size:
            append_chunk_to_h5(output_filename, traces_chunk, plaintexts_chunk, labels_chunk, rin_chunk, rout_chunk)
            # Clear lists for the next chunk
            traces_chunk, labels_chunk, plaintexts_chunk, rin_chunk, rout_chunk = [], [], [], [], []
    
    # Handle leftover traces
    if traces_chunk:
        append_chunk_to_h5(output_filename, traces_chunk, plaintexts_chunk, labels_chunk, rin_chunk, rout_chunk)

    print(f"--- Successfully created HDF5 dataset at '{output_filename}' ---")

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
    parser.add_argument("--data_out", default="./data")
    parser.add_argument("--out_format", choices=["npz, hdf5"], default="dhf5")
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
    
    dataset_base_path = os.path.join(args.data_out, f"{args.target}_{timestamp}_dataset")
    os.mkdir(dataset_base_path)

    if args.out_format == "npz":
        generate_and_save_traces(profiling_target,
                                 container,
                                 args.N_PROFILING,
                                 "profiling",
                                 args.target,
                                 timestamp)
    else:
        profiling_dataset_path = os.path.join(dataset_base_path, "profiling")
        os.mkdir(profiling_dataset_path)

        generate_h5_dataset(profiling_target,
                            container,
                            args.N_PROFILING,
                            os.path.join(profiling_dataset_path, f"profiling_{args.target}_{timestamp}.h5"))
    
    print(f"Generating {args.N_ATTACK} attack traces...")
    container = CortexMAesContainer(attack_target, args.N_ATTACK)
    print(f"{args.N_ATTACK} traces generated")

    print("Labeling attack traces...")
    k = [byte for byte in attack_target.key]
    keys = [np.array(k)] * args.N_ATTACK

    if args.out_format == "npz":
        generate_and_save_traces(attack_target,
                                container,
                                args.N_ATTACK,
                                "attack",
                                args.target,
                                timestamp)
    else:
        attack_dataset_path = os.path.join(dataset_base_path, "attack")
        os.mkdir(attack_dataset_path)
        
        generate_h5_dataset(attack_target,
                            container,
                            args.N_ATTACK,
                            os.path.join(attack_dataset_path, f"attack_{args.target}_{timestamp}.h5"))

    if args.PLOT_LEAKAGE:
        plot_dir_path = os.path.join(dataset_base_path, "plot")
        os.mkdir(plot_dir_path)
        profiling_target._plot_leakage(output_filename=os.path.join(plot_dir_path, f'annotated_trace_{args.target}_{timestamp}.png'))

    if args.CPA_ATTACK:
        print(f"Attacking traces...")
        from lascar import *
        cpa_engines = [lascar.CpaEngine(name=f"CPA_{i}", 
                                        selection_function=lambda plaintext, key_byte, index=i: sbox[plaintext[index] ^ key_byte],
                                        guess_range=range(256)) for i in range(16)]

        session = lascar.Session(CortexMAesContainer(attack_target, args.N_ATTACK), engines=cpa_engines, name="lascar CPA").run()

        guess_key = bytes([engine.finalize().max(1).argmax() for engine in cpa_engines])

        print(f"Guessed key is : {hexlify(guess_key).upper()}")