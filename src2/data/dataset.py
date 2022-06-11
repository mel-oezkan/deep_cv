import os
import numpy as np
import tensorflow as tf

from src2.data import DataGenerator
from src2.data import Processor


def create_split(split_values: tuple):
    """ Creates a tilemap of the data from which non-overlapping
    samples will be generated for the test dataset. The unused data
    will then be used for the trainng set.

    :returns: List of ids which are going to be used fro the generator
    :rtype: list
    """

    indices = np.array(
        [i[:-4] for i in os.listdir("../datasets/train/AOI_11_Rotterdam/Labels_128")])

    n_indices = len(indices)
    np.random.shuffle(indices)

    data_indices = {}
    curr_idx = 0
    for value, key in zip(split_values, ["train", "test", "validation"]):

        if value > 1:
            raise ValueError('VALUE TOOO LARGE')

        split_count = n_indices * value
        data_indices[key] = indices[curr_idx, curr_idx + split_count]

        curr_idx += split_count

    return data_indices


def create_dataset(data_name: str, in_shape: tuple, out_shape: tuple) -> tf.data.Dataset:
    """ Given a name of a dataset returns the respecitve
    dataset which will then be preprocessed
    """

    dataset_gen = None
    if data_name == "reduced_data":
        dataset_gen = DataGenerator.ResizedDataGenerator()

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

    :returns: Returns the preprocessed dataset
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
    training_data = Processor.processed_dataset(
        raw_dataset,
        data_args["batch_size"],
        im_count
    )

    return training_data
