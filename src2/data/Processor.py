import tensorflow as tf


def normalize(max, min):
    pass


def preprocess_data(dataset: tf.data.Dataset, n_counts: int, batch_size: int):

    # shuffle, batch and prefetch the dataset
    dataset = dataset.shuffle(tf.data.Autotune)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(n_counts)
    dataset = dataset.cache()

    return dataset
