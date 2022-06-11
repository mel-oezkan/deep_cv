import tensorflow as tf


def normalize(image, max: float, min: float):

    norm_im = image
    pass


def preprocess_data(dataset: tf.data.Dataset, n_counts: int, batch_size: int):

    # shuffle, batch and prefetch the dataset
    dataset = dataset.shuffle(tf.data.Autotune)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(n_counts)
    dataset = dataset.cache()

    return dataset
