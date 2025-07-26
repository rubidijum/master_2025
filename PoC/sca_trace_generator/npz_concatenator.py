import numpy as np
import os
import glob
import argparse
from collections import defaultdict
import re
import shutil

def concatenate_runs(directory):
    """
    Finds and concatenates .npz files in a directory, grouping them by
    a common run ID (target + timestamp). Saves each full run into a
    dedicated parent directory with subdirectories for profiling and attack chunks.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory not found at '{directory}'")
        return

    print(f"Scanning directory: {directory}")

    pattern = re.compile(r"^(profiling|attack)_chunk_\d+_(.+)\.npz$")
    
    # A nested dictionary to group file paths by run_id, then by prefix
    runs = defaultdict(lambda: defaultdict(list))

    # Find all chunk files and group them
    for f_path in glob.glob(os.path.join(directory, '*_chunk_*.npz')):
        basename = os.path.basename(f_path)
        match = pattern.match(basename)
        if match:
            prefix = match.group(1)
            run_id = match.group(2)
            runs[run_id][prefix].append(f_path)

    if not runs:
        print("No 'profiling_chunk' or 'attack_chunk' .npz files found to process.")
        return

    # Process each distinct run
    for run_id, prefixes in runs.items():
        # Create the main parent directory for the run
        run_output_dir = os.path.join(directory, run_id)
        os.makedirs(run_output_dir, exist_ok=True)
        print(f"\nProcessing run '{run_id}'...")
        print(f"  Main output directory created at: '{run_output_dir}'")

        # Process both 'profiling' and 'attack' prefixes if they exist for this run
        for prefix, file_list in prefixes.items():
            # The final concatenated file goes in the main run directory
            output_filename = os.path.join(run_output_dir, f"{prefix}_full.npz")
            
            print(f"  Processing {prefix} data ({len(file_list)} files)...")
            
            # Process and save the concatenated file
            process_and_save_group(sorted(file_list), output_filename)
            
            # Create the subdirectory for the chunks (e.g., profiling/, attack/)
            chunks_subdir = os.path.join(run_output_dir, prefix)
            os.makedirs(chunks_subdir, exist_ok=True)
            
            # Move the original chunk files into the new subdirectory
            print(f"  Cleaning up original {prefix} chunk files...")
            for f_path in file_list:
                try:
                    shutil.move(f_path, chunks_subdir)
                    print(f"\t* Moved {os.path.basename(f_path)} to '{prefix}/'")
                except Exception as e:
                    print(f"\t* Warning: Could not move file {os.path.basename(f_path)}. Error: {e}")


    print("\nScript finished.")

def process_and_save_group(file_list, output_filename):
    """
    Loads data from a list of .npz files, concatenates the arrays,
    and saves them to a new file.
    """
    # Dictionary to hold lists of arrays to be concatenated
    combined_data = defaultdict(list)
    # The key is constant for a run, so we only need to store it once
    key_data = None

    # Load data from each file chunk
    for f_path in file_list:
        try:
            with np.load(f_path, allow_pickle=True) as data:
                # Store the key from the first file we process
                if key_data is None and 'key' in data:
                    key_data = data['key']
                
                # Append arrays like 'traces', 'labels', 'plaintexts' to our lists
                for array_name in data.keys():
                    if array_name != 'key':
                        combined_data[array_name].append(data[array_name])
            print(f"\t* Loaded {os.path.basename(f_path)}")
        except Exception as e:
            print(f"\t* Warning: Could not process file {os.path.basename(f_path)}. Error: {e}")
            continue

    if not combined_data:
        print("No data was loaded from this group. Cannot create output file.")
        return

    # Dictionary to hold the final, concatenated arrays
    final_arrays = {}
    print("\nConcatenating arrays...")

    # Use np.vstack for robust concatenation of row-based data
    for key, arrays in combined_data.items():
        try:
            # Note: np.vstack is generally safer for this type of concatenation
            final_arrays[key] = np.vstack(arrays)
            print(f"\t* Concatenated '{key}', final shape: {final_arrays[key].shape}")
        except ValueError as e:
            print(f"\t* Warning: Could not concatenate array '{key}'. Error: {e}")
            continue
            
    # Add the single key array to our final dictionary
    if key_data is not None:
        final_arrays['key'] = key_data
    
    # Save the final combined data
    try:
        np.savez(output_filename, **final_arrays)
        print(f"\nSuccessfully saved concatenated data to '{output_filename}'")
    except Exception as e:
        print(f"\nError: Could not save the output file. Error: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Finds and concatenates chunked .npz trace files, grouping them by run.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'directory',
        type=str,
        help="The path to the directory containing the .npz files."
    )

    args = parser.parse_args()
    concatenate_runs(args.directory)