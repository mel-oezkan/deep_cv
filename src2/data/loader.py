import tensorflow as tf

from src2.data.DataGenerator import ResizedDataGenerator


def create_split():
    """Creates a tilemap of the data from which non-overlapping
    samples will be generated for the test dataset. The unused data
    will then be used for the trainng set.

    :returns: List of ids which are going to be used fro the generator
    :rtype: list
    """

    # Todo: read the geojson data and create the tilemaps

    # Todo: filter the unique values from the tilemaps and remove the indices

    pass


def create_dataset(data_name: str, in_shape: tuple, out_shape: tuple) -> tf.data.Dataset:
    """Given a name of a dataset returns the respecitve 
    dataset which will then be preprocessed
    """

    dataset_gen = None
    if data_name == "reduced_data":
        dataset_gen = ResizedDataGenerator()

    if not dataset_gen:
        raise NotImplementedError('No valid dataset is given')

    raw_dataset = tf.data.Dataset.from_generator(
        dataset_gen(),
        output_types=(tf.float32, tf.int32),
        output_shapes=(
            tf.TensorShape(in_shape),
            tf.TensorShape(out_shape)
        )
    )

    return raw_dataset, len(dataset_gen)
