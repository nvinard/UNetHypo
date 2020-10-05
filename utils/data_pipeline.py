import tensorflow as tf
from utils import data_augmentation as da
import os

# Create an input function reading a file using the Dataset API
def read_dataset(prefix, batch_size, cfg):
    def _input_fn(example_serialized):
        feature_map = {
            "x_true": tf.io.FixedLenFeature(shape=[], dtype=tf.float32, default_value=-1),
            "y_true": tf.io.FixedLenFeature(shape=[], dtype=tf.float32, default_value=-1),
            "z_true": tf.io.FixedLenFeature(shape=[], dtype=tf.float32, default_value=-1),
            "z_id": tf.io.FixedLenFeature(shape=[6], dtype=tf.float32),
            "data": tf.io.FixedLenFeature(shape=[cfg.N_TIMESAMPLES, cfg.N_TRACES], dtype=tf.float32),
        }

        parsed = tf.io.parse_single_example(example_serialized, feature_map)

        data = parsed["data"]
        # Use data only up to 96 stations not 97
        data = tf.slice(data, [0,0], [cfg.N_TIMESAMPLES, cfg.N_TRACES-1])

        # Aply data augmentations and normalize input
        data = da.tf_time_shift(data, parsed["z_id"])
        data = da.tf_gaussian_noise(data)
        data = da.tf_field_noise(data)
        data = da.tf_station_dropout(data)
        data = da.tf_normalize(data)
        data = tf.reshape(data, (cfg.N_TIMESAMPLES, cfg.N_TRACES-1, 1))

        # Define the FCN Gaussian blob label
        label = da.tf_FCN_output(parsed["x_true"], parsed["y_true"], parsed["z_true"])

        return (data, label)

    # Use prefix to create file path
    file_path = os.path.join(cfg.PATH_TO_DATA,'%s*' % prefix)

    # Create list of files that match pattern
    file_list = tf.io.matching_files(file_path)
    shards = tf.data.Dataset.from_tensor_slices(file_list)

    if prefix=="train":
        # Randomize shard file names
        shards = shards.shuffle(tf.cast(tf.shape(file_list)[0], tf.int64))
        shards = shards.repeat()

        # Feed the shards into TFRecordDataset and randomize again with interleave.
        dataset = shards.interleave(tf.data.TFRecordDataset, cycle_length=16, block_length=32, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # Shuffle the single shard file to read the data inside the shard in random order
        dataset = dataset.shuffle(buffer_size=10*batch_size)

        num_epochs = None

    elif prefix == "test":
        dataset = tf.data.TFRecordDataset(shards)
        num_epochs = 1

    dataset = dataset.map(_input_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat(num_epochs).batch(batch_size)

    return dataset.prefetch(tf.data.experimental.AUTOTUNE)
