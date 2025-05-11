import h5py
import numpy as np

metadata_map = {
    "plaintext" : 0,
    "ciphertext" : 1,
    "key" : 2,
    "masks" : 3,
    "desync" : 4
}

def get_plaintext(traces, metadata):
    """
    Extract plaintext from the traces using the metadata.
    """
    plaintext = metadata[:, metadata_map["plaintext"]]
    plaintext = np.array([np.frombuffer(plaintext[i], dtype=np.uint8) for i in range(len(plaintext))])
    return plaintext

def get_ciphertext(traces, metadata):
    """
    Extract ciphertext from the traces using the metadata.
    """
    ciphertext = metadata[:, metadata_map["ciphertext"]]
    ciphertext = np.array([np.frombuffer(ciphertext[i], dtype=np.uint8) for i in range(len(ciphertext))])
    return ciphertext

def get_key(traces, metadata):
    """
    Extract key from the traces using the metadata.
    """
    key = metadata[:, metadata_map["key"]]
    key = np.array([np.frombuffer(key[i], dtype=np.uint8) for i in range(len(key))])
    return key

def get_masks(traces, metadata):
    """
    Extract masks from the traces using the metadata.
    """
    masks = metadata[:, metadata_map["masks"]]
    masks = np.array([np.frombuffer(masks[i], dtype=np.uint8) for i in range(len(masks))])
    return masks

def get_desync(traces, metadata):
    """
    Extract desynchronization information from the metadata.
    """
    desync = metadata[:, metadata_map["desync"]]
    desync = np.array([np.frombuffer(desync[i], dtype=np.uint8) for i in range(len(desync))])
    return desync


def load_ascad(ascad_path, load_metadata=False):
    """
    Load the ASCAD dataset from the specified HDF5 file.
    """
    with h5py.File(ascad_path, 'r') as f:
        profiling_traces = np.array(f['Profiling_traces/traces'])
        profiling_labels = np.array(f['Profiling_traces/labels'])
        
        attack_traces = np.array(f['Attack_traces/traces'])
        attack_labels = np.array(f['Attack_traces/labels'])
        
        if load_metadata:
            profiling_metadata = np.array(f['Profiling_traces/metadata'])
            attack_metadata = np.array(f['Attack_traces/metadata'])
        
        if load_metadata:
            return (profiling_traces, profiling_labels, profiling_metadata), (attack_traces, attack_labels, attack_metadata)
        else:
            return (profiling_traces, profiling_labels), (attack_traces, attack_labels)
        
# TODO: dataloader for ML (keras/tensorflow)