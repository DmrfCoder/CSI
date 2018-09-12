#coding:utf-8
import tensorflow as tf

from DlTrain.Parameters import trainBatchSize, lstmTimeStep, lstmHiddenUnits, classes

'''
CNN
'''


def cnn_weight_variable(shape):
    initializer = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initializer(shape))


def cnn_bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(c_input, c_filter, c_strides):
    return tf.nn.conv2d(input=c_input, filter=c_filter, strides=c_strides, padding='VALID')


def max_pool_2x2(p_input, p_ksize, p_strides):
    return tf.nn.max_pool(value=p_input, ksize=p_ksize, strides=p_strides, padding='VALID')


'''
CNN Net
'''


def CNN(in_x):
    in_x = tf.reshape(in_x, [trainBatchSize, lstmTimeStep, lstmHiddenUnits, 1])

    w_conv1 = cnn_weight_variable([5, 5, 1, 6])
    b_conv1 = cnn_bias_variable([6])

    h_conv1 = tf.nn.relu(conv2d(c_input=in_x, c_filter=w_conv1, c_strides=[1, 1, 1, 1]) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1, [1, 2, 2, 1], [1, 2, 2, 1])

    w_conv2 = cnn_weight_variable([5, 3, 6, 10])
    b_conv2 = cnn_bias_variable([10])
    h_conv2 = tf.nn.relu(conv2d(c_input=h_pool1, c_filter=w_conv2, c_strides=[1, 3, 3, 1]) + b_conv2)

    h_pool3_flat = tf.reshape(h_conv2, [-1, 3200])  # 将32*10*10reshape为3200*1

    w_fc1 = cnn_weight_variable([3200, 1000])
    b_fc1 = cnn_bias_variable([1000])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1) + b_fc1)

    w_fc2 = cnn_weight_variable([1000, 200])
    b_fc2 = cnn_bias_variable([200])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)

    w_fc3 = cnn_weight_variable([200, classes])
    b_fc3 = cnn_bias_variable([classes])
    h_fc3 = tf.nn.relu(tf.matmul(h_fc2, w_fc3) + b_fc3)

    out_y = tf.nn.softmax(h_fc3,name='cnnSoftmax')

    return out_y


