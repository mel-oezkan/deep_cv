import tensorflow as tf


def normalize(image, max: float, min: float):
    norm_im = (image - min) / (max - min)

    return norm_im


def preprocess_data(
        dataset: tf.data.Dataset,
        n_counts: int,
        batch_size: int) -> tf.data.Dataset:
    """ Simple function that processes the tf dataset

    :param dataset: the raw dataset
    :param n_counts: number of images in the dataset
    :param batch_size: size of the batch

    :return: prefetched dataset
    :rtype: tf.data.Dataset 
    """

    # shuffle, batch and prefetch the dataset
    dataset = dataset.shuffle(n_counts)
    dataset = dataset.batch(batch_size)
    # dataset = dataset.prefetch(n_counts)

    return dataset
