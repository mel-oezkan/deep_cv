
ALL_ADAPTER_CLS = [
]


class TensorLikeDataAdapter(DataAdapter):
    """Adapter that handles Tensor-like objects, e.g. EagerTensor and NumPy."""

    @staticmethod
    def can_handle(x, y=None):
        # TODO(kaftan): Check performance implications of using a flatten
        #  here for other types of inputs.
        flat_inputs = tf.nest.flatten(x)
        if y is not None:
            flat_inputs += tf.nest.flatten(y)

        tensor_types = _get_tensor_types()

        def _is_tensor(v):
            if isinstance(v, tensor_types):
                return True
            return False

        return all(_is_tensor(v) for v in flat_inputs)
