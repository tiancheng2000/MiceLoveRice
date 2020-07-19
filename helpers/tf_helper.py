
from helpers.util import DEBUG, INFO, WARN, ERROR, hasmethod

__all__ = [
    "preload_gpu_devices",
    "async_preload_gpu_devices",
    "tf_obj_to_np_array",
    "is_tfdataset",
    "image_example",
    "norm_keep_batch_dim",
]

# IMPROVE: compatibility with TF1.x
# if tf.version.VERSION.startswith('1.'):
#     import tensorflow as tf1
# else:  # tf v2.x
#     import tensorflow.compat.v1 as tf1
#     tf1.disable_v2_behavior()

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

def async_preload_gpu_devices():
    """
    Preload in another loop/thread, hopefully call this during waiting for user inputs or other waiting period.
    """
    # IMPROVE: needn't to run in an aysncio loop (host in a new thread), to run in a new thread is enough.
    from async_ import AsyncLoop, AsyncManager

    async def coro_simple_run(): preload_gpu_devices()
    loop = AsyncManager.get_loop(AsyncLoop.DataProcess)
    DEBUG(f"[tensorflow] preload gpu devices in another thread...")
    task = AsyncManager.run_task(coro_simple_run(), loop=loop)
    return task


# TODO: make sure if this method can be replaced by `isinstance(obj, tf.data.Dataset)` in every case
def is_tfdataset(obj):
    import tensorflow as tf
    # ds = tf.data.Dataset.from_tensor_slices([''])  # => tensorflow.python.data.ops.dataset_ops.TensorSliceDataset
    # ds = ds.map(lambda x: x)  # => tensorflow.python.data.ops.dataset_ops.MapDataset
    # return type(obj).__name__.endswith('Dataset')
    return isinstance(obj, tf.data.Dataset)  # *Dataset(*={Zip,Map,Take,Skip..}) are all Database

def tf_obj_to_np_array(tf_obj: object):
    """
    NOTE: for big tf.Dataset, such conversion, which flush all into memory, will exhaust resources.
    """
    import tensorflow as tf
    import numpy as np
    # assert isinstance(tf_data, tf.data.Dataset)  # NOTE: MapDataset, or Prefetch/Take/.. is NOT Dataset...
    if hasattr(tf_obj, 'element_spec') and type(tf_obj.element_spec) is tf.TensorSpec:
        assert hasmethod(tf_obj, 'as_numpy_iterator'), 'tf dataset must have `as_numpy_iterator` for conversion'
        return np.array(list(tf_obj.as_numpy_iterator()))
    elif hasmethod(tf_obj, 'numpy'):
        return tf_obj.numpy()


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    import tensorflow as tf
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()   # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    import tensorflow as tf
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    import tensorflow as tf
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def image_example(image_string, path=None, label=None):
    import tensorflow as tf
    image_t = tf.image.decode_jpeg(image_string)
    feature = {
        'h': _int64_feature(image_t.shape[0]),
        'w': _int64_feature(image_t.shape[1]),
        'c': _int64_feature(image_t.shape[2]),
        'path': _bytes_feature(tf.constant(path, dtype=tf.dtypes.string)),  # IMPROVE: .encode('utf-8')
        'label': _int64_feature(label),
        'image_t': _float_feature(image_t.numpy().flatten()),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def norm_keep_batch_dim(np_or_tf_arr):
    """
    :param np_or_tf_arr: (batch=1, width, height, feature), a 4D np.ndarray or tf.Tensor
    :return:
    """
    import numpy as np
    if isinstance(np_or_tf_arr, np.ndarray):
        for i in range(np_or_tf_arr.shape[0]):  #
            np_or_tf_arr[i] = np_or_tf_arr[i] / np.linalg.norm(np_or_tf_arr[i])
    else:
        import tensorflow as tf
        if isinstance(np_or_tf_arr, tf.Tensor):
            for i in range(np_or_tf_arr.shape[0]):
                np_or_tf_arr[i] = np_or_tf_arr[i] / tf.linalg.norm(np_or_tf_arr[i])
    return np_or_tf_arr
