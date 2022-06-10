import tensorflow as tf
from src2.data.loader import create_dataset

# Todo: is this function usefull


def convert_params(params: dict) -> dict:
    """Given a list of dictionaries converts the 
    arguments into a large dictionary

    :param params: Contains the parameters for the data pipeline
    :type: list

    :returns: Dictionary of combined arguments
    :rtype: dict
    """
    return params


def load_data(data_args: dict) -> tf.data.Dataset:
    """Fetch all parameters and load the respective dataset

    :params data_args: contains all the variabels for the data pipline
    :type: dict

    :returns: Returns the preprocessed dataset
    :rtype: tf.data.Dataset
    """

    # initalize the split

    # initalize the generator and create the dataset
    dataset = create_dataset()

    return
