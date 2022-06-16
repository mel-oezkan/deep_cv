import tensorflow as tf


def init_environment(env_params: dict):
    tf.keras.backend.clear_session()
    limit_gpu(env_params.get('gpu_limit'))


def limit_gpu(limit: int):
    if not limit:
        return

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(
                    memory_limit=limit)]
            )
        except RuntimeError as e:
            print(e)
