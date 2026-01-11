import h5py
import numpy as np
import tensorflow as tf
import os

from tqdm import tqdm

from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.layers import (
    Activation,
    Add,
    AveragePooling1D,
    BatchNormalization,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Input,
    Layer,
    MaxPooling1D,
    Multiply,
    Reshape
)

from tuning_utils import CleanMemoryTuner, model_builder_synthetic_se_cnn, vgg_cbam_synthetic_builder
from models import cbam_block, vgg_cnn_cbam
from sca_utils import SBOX, run_rank_trials, plot_mean_rank

import h5py
import numpy as np
import tensorflow as tf
import os

class SyntheticDataLoader():

    def __init__(self, file_path):
        self.file_path = file_path

    def data_generator(self, window_index=0):
        """
        A generator that yields traces, labels, and metadata (plaintext, key)
        one by one from an HDF5 file.
        """
        with h5py.File(self.file_path, 'r') as hf:
            traces = hf['windows']
            labels = hf['labels']
            plaintexts = hf['metadata/plaintexts']
            keys = hf['metadata/key']
            num_samples = traces.shape[0]

            is_key_scalar =(keys.shape == ())
            if is_key_scalar:
                fixed_key_value = keys[()]
            else:
                is_key_fixed = (keys.ndim == 1 and keys.shape[0] != num_samples)
                if is_key_fixed:
                    fixed_key_value = keys[:]

            for i in range(num_samples):
                trace = traces[i, window_index, :].astype('float32')
                trace = np.expand_dims(trace, axis=-1)
                label = labels[i].astype('int32')
                pt = plaintexts[i]

                if is_key_scalar or is_key_fixed:
                    key = fixed_key_value
                else:
                    key = keys[i]

                yield trace, label, pt, key

    def create_dataset(self, window_index=0, batch_size=64, val_split=0.2):
        """
        Creates and splits a tf.data.Dataset for training and validation.
        """
        with h5py.File(self.file_path, 'r') as hf:
            num_samples = hf['windows'].shape[0]
            input_dim = hf['windows'].shape[2]

            pt_dset = hf['metadata/plaintexts']
            pt_shape = (pt_dset.shape[1],) if pt_dset.ndim > 1 else ()

            key_dset = hf['metadata/key']
            
            # A key element is a vector if the dataset is 2D OR if it's a fixed key (1D but not length num_samples).
            if key_dset.shape == ():
                key_val = key_dset[()]
                key_shape = key_val.shape if isinstance(key_val, np.ndarray) else ()
            elif key_dset.ndim > 1 or (key_dset.ndim == 1 and key_dset.shape[0] != num_samples):
                key_shape = (key_dset.shape[-1],)
            else:
                key_shape = ()
            

        dataset = tf.data.Dataset.from_generator(
            lambda: self.data_generator(window_index),
            output_signature=(
                tf.TensorSpec(shape=(input_dim, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=pt_shape, dtype=tf.uint8),
                # MODIFIED: Use the new, more robust key_shape.
                tf.TensorSpec(shape=key_shape, dtype=tf.uint8)
            )
        )

        val_size = int(num_samples * val_split)
        if val_size < batch_size:
            print("[WARN] Validation set smaller than batch size! Increase dataset size or reduce batch size.")
        train_size = num_samples - val_size

        if train_size > 0:
            dataset = dataset.shuffle(buffer_size=min(train_size, 10000))

        ds_full_train = dataset.take(train_size)
        ds_full_val = dataset.skip(train_size)

        ds_train = ds_full_train.map(lambda trace, label, pt, key: (trace, label))

        ds_train = ds_train.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
        ds_full_val_batched = ds_full_val.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

        print(f"Dataset created from '{os.path.basename(self.file_path)}':")
        print(f"  - Total samples: {num_samples}")
        print(f"  - Training samples: {train_size}")
        print(f"  - Validation samples: {val_size}")

        return ds_train, ds_full_val_batched, input_dim, train_size, val_size

class HdfToTfrAdapter():
  def _bytes_feature(self, value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  def _int64_feature(self, value):
      """Returns an int64_list from a bool / enum / int / uint."""
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

  def serialize_example(self, trace, label, plaintext, key_byte):
      """
      Creates a tf.train.Example message ready to be written to a file.
      (MODIFIED to use .tobytes() for stability)
      """
      feature = {
          'trace': self._bytes_feature(trace.tobytes()),
          'label': self._int64_feature(label),
          'plaintext': self._int64_feature(plaintext),
          'key_byte': self._int64_feature(key_byte),
      }
      example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
      return example_proto.SerializeToString()

  def convert_h5_to_tfrecord(self, h5_path, tfrecord_path, window_index=0):
      """
      Converts a single HDF5 file to the TFRecord format, including metadata.
      This version handles cases where metadata might be a scalar value.
      """
      print(f"Starting conversion: {h5_path} -> {tfrecord_path}")

      with h5py.File(h5_path, 'r') as hf:
          # Load all necessary datasets
          traces = hf['windows']
          labels = hf['labels']
          plaintexts_dset = hf['metadata/plaintexts']
          keys_dset = hf['metadata/key']
          num_samples = traces.shape[0]

          # Check if metadata datasets are scalar (single value) or arrays
          is_plaintext_scalar = plaintexts_dset.ndim == 0
          is_key_scalar = keys_dset.ndim == 0

          # If a dataset is scalar, read its single value once.
          plaintext_val = plaintexts_dset[()] if is_plaintext_scalar else None
          key_val = keys_dset[()] if is_key_scalar else None

          with tf.io.TFRecordWriter(tfrecord_path) as writer:
              for i in tqdm(range(num_samples), desc=f"Converting {os.path.basename(h5_path)}"):
                  # Extract the specific window and label
                  trace_data = traces[i, window_index, :].astype('float32')
                  label_data = labels[i].astype('int32')

                  # Get the metadata value: use the single scalar value or slice the array
                  plaintext_data = plaintext_val if is_plaintext_scalar else plaintexts_dset[i]
                  key_byte_data = key_val if is_key_scalar else keys_dset[i]

                  # Serialize and write the complete record to the file
                  example = self.serialize_example(trace_data, label_data, int(plaintext_data), int(key_byte_data))
                  writer.write(example)

      print(f"Conversion complete for {tfrecord_path}")

  def convert_bytes(self, input_dir, bytes_to_convert, output_dir, window_to_use=0):
    os.makedirs(output_dir, exist_ok=True)

    for target_byte in bytes_to_convert:
        h5_file = os.path.join(input_dir, f'byte_{target_byte}.h5')
        tfrecord_file = os.path.join(output_dir, f'byte_{target_byte}.tfrecord')

        if not os.path.exists(h5_file):
            print(f"Dummy H5 file for byte {target_byte} not found. Creating it.")
            with h5py.File(h5_file, 'w') as f:
                f.create_dataset('windows', data=np.random.rand(50000, 5, 701).astype(np.float32))
                f.create_dataset('labels', data=np.random.randint(0, 256, size=50000).astype(np.int32))
                f.create_dataset('metadata/plaintexts', data=np.random.randint(0, 256, size=50000).astype(np.uint8))
                f.create_dataset('metadata/key', data=np.random.randint(0, 256, size=50000).astype(np.uint8))

        self.convert_h5_to_tfrecord(h5_file, tfrecord_file, window_to_use)

class TfrDataLoader():
    def serialize_example(self, trace, label, plaintext, key_byte):
      """
      Creates a tf.train.Example message ready to be written to a file.
      """
      feature = {
          'trace': _bytes_feature(trace.tobytes()),
          'label': _int64_feature(label),
          'plaintext': _int64_feature(plaintext),
          'key_byte': _int64_feature(key_byte),
      }
      example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
      return example_proto.SerializeToString()

    def _parse_tfrecord_fn(self, example_proto, input_dim=701):
      """
      Parses a single record from a TFRecord file.
      """
      feature_description = {
          'trace': tf.io.FixedLenFeature([], tf.string),
          'label': tf.io.FixedLenFeature([], tf.int64),
          'plaintext': tf.io.FixedLenFeature([], tf.int64),
          'key_byte': tf.io.FixedLenFeature([], tf.int64),
      }
      example = tf.io.parse_single_example(example_proto, feature_description)

      trace = tf.io.decode_raw(example['trace'], out_type=tf.float32)
      trace = tf.reshape(trace, [input_dim, 1])

      label = tf.cast(example['label'], tf.int32)
      plaintext = tf.cast(example['plaintext'], tf.uint8)
      key_byte = tf.cast(example['key_byte'], tf.uint8)

      return trace, label, plaintext, key_byte

    def create_dataset_from_tfrecord(self, tfrecord_path, input_dim, batch_size, val_split=0.2, num_samples=50000):
        # Load and parse dataset first
        raw_ds = tf.data.TFRecordDataset(tfrecord_path, num_parallel_reads=tf.data.AUTOTUNE)
        parsed_ds = raw_ds.map(lambda x: self._parse_tfrecord_fn(x, input_dim=input_dim), num_parallel_calls=tf.data.AUTOTUNE)
    
        # Count actual number of records
        actual_count = sum(1 for _ in parsed_ds)
        print(f"[INFO] Actual number of records in TFRecord: {actual_count}")
    
        # Clamp num_samples if necessary
        num_samples = min(actual_count, num_samples)
        val_size = int(val_split * num_samples)
        train_size = num_samples - val_size
        print(f"[INFO] Using {train_size} training and {val_size} validation samples")
    
        # Reset and re-map dataset (first pass exhausted it)
        raw_ds = tf.data.TFRecordDataset(tfrecord_path, num_parallel_reads=tf.data.AUTOTUNE)
        parsed_ds = raw_ds.map(lambda x: self._parse_tfrecord_fn(x, input_dim=input_dim), num_parallel_calls=tf.data.AUTOTUNE)
    
        parsed_ds = parsed_ds.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
        parsed_ds = parsed_ds.take(num_samples)
    
        # Now split
        ds_train_full = parsed_ds.take(train_size)
        ds_val_full = parsed_ds.skip(train_size)
    
        # For tuner: only trace and label
        ds_train_for_tuner = ds_train_full.map(lambda trace, label, pt, key: (trace, label))
        ds_val_for_tuner = ds_val_full.map(lambda trace, label, pt, key: (trace, label))
    
        # Batch safely (don’t drop small val batches!)
        ds_train_for_tuner = ds_train_for_tuner.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
        ds_val_for_tuner = ds_val_for_tuner.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    
        print("Train batches:", sum(1 for _ in ds_train_for_tuner))
        print("Val batches:", sum(1 for _ in ds_val_for_tuner))
    
        return ds_train_for_tuner, ds_val_for_tuner, ds_val_full, train_size, val_size
