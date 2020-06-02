import os.path as osp

import tensorflow as tf

__all__ = [
    "decode_image_file",
    "decode_integer_label",
]


# TODO: allow Sequential decode functions
@tf.function
def decode_image_file(path, encoding='jpg', colormode=None, reshape: list = None,
                      preserve_aspect_ratio=True, color_transform=None, normalize=True) -> tf.Tensor:
    """
    Process: path -> Tensor(string) -> Tensor(int(w,h,c)) -> 3-channel -> float32 -> resize -> normalize
    :param path: Tensor of string
    :param encoding:
    :param colormode: 'grayscale', 'rgb', 'rgba', None(as is). ref:tf.keras.preprocessing.image.load_img()
    :param reshape: [b, h, w, c], b will be ignored
    :param preserve_aspect_ratio:
    :param color_transform:
    :param normalize:
    :return:
    """
    image = tf.io.read_file(path, name="decode_image_file.input")

    # NOTE: cannot use dict.get() in an autograph, dict is not `Hashable`
    # channels = {'grayscale': 1, 'rgb': 3, 'rgba': 4}.get(colormode, 0)
    if colormode == 'grayscale':
        channels = 1
    elif colormode == 'rgb':
        channels = 3
    elif colormode == 'rgba':
        channels = 4
    else:
        channels = 0
    # NOTE: 只支持每次解码一个（因为h, w, c未必能对齐）
    # NOTE: tf.decode_返回[h, w, c]（除了 gif，在decode_image()且expand_animations=True时返回[num_frames,..]）
    if encoding == 'jpg':
        image = tf.io.decode_jpeg(image, channels=channels)
    elif encoding == 'png':
        image = tf.io.decode_png(image, channels=channels)
    elif encoding == 'bmp':
        image = tf.io.decode_bmp(image, channels=channels)
    elif encoding == 'gif':
        image = tf.io.decode_gif(image)
    else:
        # NOTE: tf.image.decode_image when used in tf.data will return no shape info..because no eagerly executed
        #   ref: https://github.com/tensorflow/tensorflow/issues/28247
        # image = tf.io.decode_image(image, channels=1 if colormode == 'grayscale' else 0, expand_animations=False)
        # image.set_shape([None, None, None])
        # IMPROVE: determine encoding by file_ext or file header
        raise ValueError(f"Unsupported image encoding: {encoding}")

    # TODO: move the following steps to model_fn for batch processing
    # NOTE: TF2.x can handle with flexible Input, includin variable channel dim
    # if colormode == 'grayscale':
    #     image = tf.image.grayscale_to_rgb(image)  # channel: 1->3
    # FIXME: why ValueError('\'size\' must be a 1-D int32 Tensor') even when preserve_aspect_ratio is True
    # if (resize_w is not None and resize_h is not None) \
    #         or (preserve_aspect_ratio and (resize_w, resize_h) != (None, None)):
    if reshape is not None:
        # NOTE: 记得图像处理、DL领域一直用(height, width, channel)这一顺序
        _, resize_h, resize_w, _ = reshape
        image = tf.image.resize(image, [resize_h, resize_w], preserve_aspect_ratio=preserve_aspect_ratio)  # bilinear
    if color_transform == "complementary":
        image = tf.math.subtract(tf.constant(255, dtype=image.dtype), image)
    if normalize:
        image = tf.cast(image, tf.float32)
        # IMPROVE: consider use true mean and stddev
        image = tf.divide(tf.subtract(image, [0.0]), [255.0])  # normalize
    return image


def decode_integer_label(label):
    label = tf.strings.to_number(label, tf.dtypes.int32)
    # label = tf.reshape(label, [])  # label is a scalar
    return label
