import tensorflow as tf


def exponential(x):
    return 1.0 * (tf.math.exp(x) - 1.0)
