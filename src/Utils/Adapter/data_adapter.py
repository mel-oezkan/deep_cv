import abc
import tensorflow as tf

from src.Utils.Adapter.utils import _is_distributed_dataset, is_none_or_empty


# source: https://github.com/keras-team/keras/blob/07e13740fd181fc3ddec7d9a594d8a08666645f6/keras/engine/data_adapter.py#L980

class DataAdapter(object, metaclass=abc.ABCMeta):
    """Base class for input data adapter.
    In TF 2.0, tf.data is the preferred API for user to feed in data. In order
    to simplify the training code path, all the input data object will be
    converted to `tf.data.Dataset` if possible.
    Note that since this class is mainly targeted for TF 2.0, it might have a lot
    of assumptions under the hood, eg eager context by default, distribution
    strategy, etc. In the meantime, some legacy feature support might be dropped,
    eg, Iterator from dataset API in v1, etc.
    The sample usage of this class is like:
    ```
    x = tf.data.Dataset.range(100)
    adapter_cls = [NumpyArrayDataAdapter, ..., DatasetAdapter]
    applicable_adapters = [cls for cls in adapter_cls if cls.can_handle(x)]
    if len(applicable_adapters) != 1:
      raise ValueError("Expect only one adapter class to handle the input")
    dataset = applicable_adapters[0](x).get_dataset()
    for data in dataset:
      # training
    ```
    """

    @staticmethod
    def can_handle(x, y=None):
        """Whether the current DataAdapter could handle the input x and y.
        Structure wise, x and y can be single object, or list of objects if there
        multiple input/output, or dictionary of objects when the input/output are
        named.
        Args:
          x: input features.
          y: target labels. Note that y could be None in the case of prediction.
        Returns:
          boolean
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __init__(self, x, y=None, **kwargs):
        """Create a DataAdapter based on data inputs.
        The caller must make sure to call `can_handle()` first before invoking this
        method. Provide unsupported data type will result into unexpected behavior.
        Args:
          x: input features.
          y: target labels. Note that y could be None in the case of prediction.
          **kwargs: Other keyword arguments for DataAdapter during the construction
            of the tf.dataset.Dataset. For example:
            - Numpy data might have `sample_weights` which will be used for
              weighting the loss function during training.
            - Numpy data might need to have `batch_size` parameter when constructing
              the dataset and iterator.
            - Certain input might need to be distribution strategy aware. When
              `distribution_strategy` is passed, the created dataset need to respect
              the strategy.
            DataAdapter might choose to ignore any keyword argument if it doesn't
            use it, or raise exception if any required argument is not provide.
        """
        if not self.can_handle(x, y):
            raise ValueError("{} Cannot handle input {}, {}".format(
                self.__class__, x, y))

    @abc.abstractmethod
    def get_dataset(self):
        """Get a dataset instance for the current DataAdapter.
        Note that the dataset returned does not repeat for epoch, so caller might
        need to create new iterator for the same dataset at the beginning of the
        epoch. This behavior might change in future.
        Returns:
          An tf.dataset.Dataset. Caller might use the dataset in different
          context, eg iter(dataset) in eager to get the value directly, or in graph
          mode, provide the iterator tensor to Keras model function.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_size(self):
        """Return the size (number of batches) for the dataset created.
        For certain type of the data input, the number of batches is known, eg for
        Numpy data, the size is same as (number_of_element / batch_size). Whereas
        for dataset or python generator, the size is unknown since it may or may not
        have a end state.
        Returns:
          int, the number of batches for the dataset, or None if it is unknown. The
          caller could use this to control the loop of training, show progress bar,
          or handle unexpected StopIteration error.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def batch_size(self):
        """Return the batch size of the dataset created.
        For certain type of the data input, the batch size is known, and even
        required, like numpy array. Where as for dataset, the batch is unknown
        unless we take a peek.
        Returns:
          int, the batch size of the dataset, or None if it is unknown.
        """
        raise NotImplementedError

    def representative_batch_size(self):
        """Return a representative size for batches in the dataset.
        This is not guaranteed to be the batch size for all batches in the
        dataset. It just needs to be a rough approximation for batch sizes in
        the dataset.
        Returns:
          int, a representative size for batches found in the dataset,
          or None if it is unknown.
        """
        return self.batch_size()

    @abc.abstractmethod
    def has_partial_batch(self):
        """Whether the dataset has partial batch at the end."""
        raise NotImplementedError

    @abc.abstractmethod
    def partial_batch_size(self):
        """The size of the final partial batch for dataset.
        Will return None if has_partial_batch is False or batch_size is None.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def should_recreate_iterator(self):
        """Returns whether a new iterator should be created every epoch."""
        raise NotImplementedError

    def get_samples(self):
        """Returns number of samples in the data, or `None`."""
        if not self.get_size() or not self.batch_size():
            return None
        total_sample = self.get_size() * self.batch_size()
        if self.has_partial_batch():
            total_sample -= (self.batch_size() - self.partial_batch_size())
        return total_sample

    def on_epoch_end(self):
        """A hook called after each epoch."""
        pass


class DatasetAdapter(DataAdapter):
    """Adapter that handles `tf.data.Dataset`."""

    @staticmethod
    def can_handle(x, y=None):
        return (isinstance(x, (tf.compat.v1.data.Dataset, tf.data.Dataset)) or
                _is_distributed_dataset(x))

    def __init__(self,
                 x,
                 y=None,
                 sample_weights=None,
                 steps=None,
                 **kwargs):
        super(DatasetAdapter, self).__init__(x, y, **kwargs)
        # Note that the dataset instance is immutable, its fine to reuse the user
        # provided dataset.
        self._dataset = x

        # The user-provided steps.
        self._user_steps = steps

        self._validate_args(y, sample_weights, steps)

    def get_dataset(self):
        return self._dataset

    def get_size(self):
        return  # Inferred in `DataHandler`.

    def batch_size(self):
        return None

    def has_partial_batch(self):
        return False

    def partial_batch_size(self):
        return None

    def should_recreate_iterator(self):
        # Since DistributedDatasets have no cardinality, the user must provide
        # all steps that need to be run, calling `.repeat()` as needed.
        if _is_distributed_dataset(self._dataset):
            return False

        # If user doesn't supply `steps`, or if they supply `steps` that
        # exactly equals the size of the `Dataset`, create a new iterator
        # each epoch.
        return (self._user_steps is None or
                tf.data.experimental.cardinality(self._dataset).numpy() == self._user_steps)

    def _validate_args(self, y, sample_weights, steps):
        """Validates `__init__` arguments."""
        # Arguments that shouldn't be passed.
        if not is_none_or_empty(y):
            raise ValueError("`y` argument is not supported when using "
                             "dataset as input.")
        if not is_none_or_empty(sample_weights):
            raise ValueError("`sample_weight` argument is not supported when using "
                             "dataset as input.")

        if steps is None:
            if _is_distributed_dataset(self._dataset):
                raise ValueError("When providing a distributed dataset, you must "
                                 "specify the number of steps to run.")

            size = tf.data.experimental.cardinality(self._dataset).numpy()
            if size == tf.data.experimental.INFINITE_CARDINALITY and steps is None:
                raise ValueError(
                    "When providing an infinite dataset, you must specify "
                    "the number of steps to run (if you did not intend to "
                    "create an infinite dataset, make sure to not call "
                    "`repeat()` on the dataset).")
