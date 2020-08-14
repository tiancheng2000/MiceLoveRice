
__all__ = [
    "AdaIN",
]

def AdaIN(content_features, style_features, alpha=1, epsilon=1e-5):
    """
    Used in Style Transfer (2020/08).
    Normalizes the `content_features` with scaling and offset from `style_features`.
    :param content_features: features retrieved from content images
    :param style_features: features retrieved from style images
    :param alpha: a value smaller than 1.0 will keep more content and convert less style
    :param epsilon: a small float to avoid dividing by 0, used by `tf.nn.batch_normalization()`
    :return: normalized features, which can be used to generate new images.
    """
    import tensorflow as tf
    # UPDATE: keep_dims -> keepdims
    content_mean, content_variance = tf.nn.moments(content_features, [1, 2], keepdims=True)
    style_mean, style_variance = tf.nn.moments(style_features, [1, 2], keepdims=True)

    normalized_content_features = tf.nn.batch_normalization(
        content_features, content_mean, content_variance, style_mean, tf.sqrt(style_variance), epsilon
    )
    normalized_content_features = alpha * normalized_content_features + (1 - alpha) * content_features
    return normalized_content_features
