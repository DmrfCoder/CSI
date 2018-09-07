import tensorflow as tf


def cnn_weight_variable(shape):
    initializer = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initializer(shape))


def cnn_bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, f, s):
    return tf.nn.conv2d(input=x, filter=f, strides=s, padding='SAME')


def max_pool_2x2(x, k, s):
    return tf.nn.max_pool(value=x, ksize=k, strides=s, padding='SAME')