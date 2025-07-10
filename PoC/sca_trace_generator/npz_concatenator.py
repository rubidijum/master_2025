import numpy as np
import os
import glob
import argparse
from collections import defaultdict

def concatenate_npz_files(directory):
    """
    Finds and concatenates .npz files in a given directory, grouping them by name.

    It looks for files matching 'profiling_chunk_*.npz' and 'attack_chunk_*.npz',
    concatenates the arrays within them, and saves the results to
    'profiling_full.npz' and 'attack_full.npz'.

    Args:
        directory (str): The path to the directory containing the .npz files.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory not found at '{directory}'")
        return

    print(f"Scanning directory: {directory}")

    # Use glob to find all files matching the patterns
    profiling_files = sorted(glob.glob(os.path.join(directory, 'profiling_chunk_*.npz')))
    attack_files = sorted(glob.glob(os.path.join(directory, 'attack_chunk_*.npz')))

    if not profiling_files and not attack_files:
        print("No 'profiling_chunk' or 'attack_chunk' .npz files found to process.")
        return

    # Process profiling files
    if profiling_files:
        print(f"\nFound {len(profiling_files)} profiling files. Processing...")
        process_and_save_group(profiling_files, os.path.join(directory, 'profiling_full.npz'))
    else:
        print("\nNo profiling files found.")

    # Process attack files
    if attack_files:
        print(f"\nFound {len(attack_files)} attack files. Processing...")
        process_and_save_group(attack_files, os.path.join(directory, 'attack_full.npz'))
    else:
        print("\nNo attack files found.")

    print("\nScript finished.")

def process_and_save_group(file_list, output_filename):
    """
    Loads data from a list of .npz files, concatenates the arrays, and saves to a new file.

    Args:
        file_list (list): A list of paths to .npz files.
        output_filename (str): The path for the output .npz file.
    """
    # A dictionary to hold lists of arrays, with keys corresponding to the array names in the .npz files
    combined_data = defaultdict(list)
    # Hold constant array - key
    constant_data = {}
    is_first_file = True

    # Load data from each file and group arrays by their key
    for f_path in file_list:
        try:
            with np.load(f_path) as data:
                for key in data.keys():
                    if is_first_file:
                        if 'key' in key.lower():
                            constant_data[key] = data[key]
                        else:
                            combined_data[key].append(data[key])
                    elif 'key' not in key.lower():
                        combined_data[key].append(data[key])
            print(f"\t* Loaded {os.path.basename(f_path)}")
            is_first_file = False
        except Exception as e:
            print(f"\t* Warning: Could not process file {os.path.basename(f_path)}. Error: {e}")
            continue

    if not combined_data and not constant_data:
        print("No data was loaded. Cannot create output file.")
        return

    # A dictionary to hold the final concatenated arrays
    final_arrays = {}
    print("\nConcatenating arrays...")

    # Concatenate arrays for each key
    for key, arrays in combined_data.items():
        try:
            final_arrays[key] = np.concatenate(arrays)
            print(f"\t* Concatenated array '{key}' with final shape: {final_arrays[key].shape}")
        except ValueError as e:
            print(f"\t* Warning: Could not concatenate array '{key}'. Error: {e}")
            print("\t  This can happen if the arrays have incompatible shapes.")
            continue

    # Replicate the constant key to match the concatenated data shape
    target_length = 0
    if final_arrays:
        target_length = final_arrays['traces'].shape[0]

    if target_length > 0:
        print(f"\nReplicating constant key {target_length} times...")
        replicated_arr = np.tile(constant_data['key'], (target_length, 1))
        final_arrays['key'] = replicated_arr
    else:
        print("\t  No key to replicate, check the input files")
    
    # Save the concatenated arrays to a new .npz file
    try:
        np.savez(output_filename, **final_arrays)
        print(f"\nSuccessfully saved concatenated data to '{output_filename}'")
    except Exception as e:
        print(f"\nError: Could not save the output file. Error: {e}")


if __name__ == '__main__':
    # Set up the command-line argument parser
    parser = argparse.ArgumentParser(
        description="Concatenate numpy (.npz) files in a directory.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'directory',
        type=str,
        help="The path to the directory containing the .npz files."
    )

    args = parser.parse_args()
    
    # Run the main function
    concatenate_npz_files(args.directory)