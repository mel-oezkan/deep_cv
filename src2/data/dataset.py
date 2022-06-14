import tensorflow as tf

from src2.data.DataGenerator import ResizedDataGenerator
from src2.data.split_tiles import split_tiles, recombine_splits
from src2.data.preprocessing import preprocess_data


def create_split(split_proportions={'train':0.7, 'valid':0.15, 'test':0.15},
                 strip_number=10, geojson_name='tile_positions.geojson'):
    """Create dataset splits. These are minimally overlapping and roughly have
    the proportion provided with split_proportions.

    :param split_proportions: names of different datasets and the proportion
        of their elements, defaults to {'train':0.7, 'valid':0.15, 'test':0.15}
    :type split_proportions: dictonary of floats, optional
    :param strip_number: number of strips to cut map of tiles into, the more
        strips closer the number of ids in the returned splits to their target
        number. This is because these n minimally overlapping strips are
        recombined to create the final splits, defaults to 10
    :type strip_number: integer, optional
    :param geojson_name: name of geojson file that is created in the process,
        defaults to tile_positions.geojson
    :type geojson_name: string with ending .geojson, optional
    :returns: dictonary with an array of ids for each split
    :rtype: dictonary with np.ndarray's as values
    """
    cuts = split_tiles(geojson_name, strip_number)
    splits = recombine_splits(cuts, split_proportions)
    return splits


def create_dataset(data_name: str, in_shape: tuple, out_shape: tuple) -> tf.data.Dataset:
    """ Given a name of a dataset returns the respecitve 
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


def load_data(data_args: dict) -> tf.data.Dataset:
    """Fetch all parameters and load the respective dataset

    :params data_args: contains all the variabels for the data pipline
    :type: dict
    :return: Returns the preprocessed dataset
    :rtype: tf.data.Dataset
    """

    # initalize the split

    # initalize the generator and create the dataset
    print('Creating raw dataset')
    raw_dataset, im_count = create_dataset(
        data_args["dataset_name"],
        data_args["in_shape"],
        data_args["out_shape"],
    )

    # print processing raw dataset
    training_data = preprocess_data(
        raw_dataset,
        data_args["batch_size"],
        im_count
    )

    return training_data
