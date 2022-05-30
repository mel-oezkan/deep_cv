
import tensorflow as tf


def _is_distributed_dataset(ds):
    return isinstance(ds, tf.distribute.DistributedDataset)


def is_none_or_empty(inputs):
    # util method to check if the input is a None or a empty list.
    # the python "not" check will raise an error like below if the input is a
    # numpy array
    # "The truth value of an array with more than one element is ambiguous.
    # Use a.any() or a.all()"
    return inputs is None or not tf.nest.flatten(inputs)
