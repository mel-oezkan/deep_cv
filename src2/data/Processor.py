import tensorflow as tf


def preprocess_data(dataset, batch_size, n_counts):

    # shuffle, batch and prefetch the dataset
    dataset = dataset.shuffle(tf.data.Autotune)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(n_counts)
    dataset = dataset.cache()

    return dataset
