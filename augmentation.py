import tensorflow as tf


def horizontal_flip(x):

    return tf.image.random_flip_left_right(x)


def mix_up(x, y, beta):

    mix = tf.distributions.Beta(beta, beta).sample([tf.shape(x)[0], 1, 1, 1])
    mix = tf.maximum(mix, 1 - mix)
    xmix = x * mix + x[::-1] * (1 - mix)
    lmix = y * mix[:, :, 0, 0] + y[::-1] * (1 - mix[:, :, 0, 0])
    return xmix, lmix


def random_translation(x, w):

    y = tf.pad(x, [[0] * 2, [w] * 2, [w] * 2, [0] * 2], mode='REFLECT')
    return tf.random_crop(y, tf.shape(x))
