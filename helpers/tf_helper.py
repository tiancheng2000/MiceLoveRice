
from helpers.util import DEBUG, INFO, WARN, ERROR

__all__ = [
    "preload_gpu_devices",
    "tf_data_to_np_array",
]

__preloaded_gpu___ = False

def preload_gpu_devices():
    global __preloaded_gpu___
    if __preloaded_gpu___:
        return
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    __preloaded_gpu___ = True
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            INFO("Physical GPU Memory Growth is turned ON.")
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            INFO(f"Num of Physical GPUs: {len(gpus)}, Num of Logical GPU: {len(logical_gpus)}")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            ERROR(f"Exception during preload_gpu_devices: {e}")
    else:
        WARN("No physical GPU available.")

def tf_data_to_np_array(tf_data: object):
    import tensorflow as tf
    import numpy as np
    assert isinstance(tf_data, tf.data.Dataset)
    if type(tf_data.element_spec) is tf.TensorSpec:
        return np.array(list(tf_data.as_numpy_iterator()))
    else:
        raise ValueError(f"element_spec must be TensorSpec for conversion whereas {type(tf_data.element_spec)}")