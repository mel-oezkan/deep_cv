import tensorflow as tf

from src.Data.DataGenerator import ResizedDataGenerator


def load_data(data_name: str, amount=None) -> tf.data.Dataset:
    """Function to load the given dataset and apply preprocessing steps.

    :param data_name: string mapping of the dataset
    :type: string
    :param amount: optinal value to limit the datapoints
    :type: int

    :returns: The preprocessed dataset
    :rtype: tf.data.Dataset
    """

    dataset_gen = None
    if data_name == "reduced":
        dataset_gen = small_data(amount)

    if not dataset_gen:
        raise NotImplementedError('No valid dataset is given')

    dataset = tf.data.Dataset.from_generator(
        dataset_gen(),
        output_types=(tf.float32, tf.int32),
        output_shapes=(
            tf.TensorShape([128, 128, 4]),
            tf.TensorShape([128, 128, 1])
        )
    )

    return dataset


def preprocess_data(data, options=None):

    # shuffle, batch and prefetch the dataset
    dataset = dataset.shuffle(10_000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch_size)
    
    pass


def small_data(amount) -> tuple:
    """ Custom function that handles the data input pipeline
    for the small dataset (resized dataset to 128x128)

    """

    im_generator = ResizedDataGenerator()
