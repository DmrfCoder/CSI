# -*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

from Util.ReadAndDecodeUtil import read_and_decode

classes = 5

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
    # in_x=tf.reshape(in_x, [-1, 200, 64, 1])
    in_x = tf.reshape(in_x, [train_batch, lstm_time_step, lstm_hidden_units, 1])

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

    out_y = tf.nn.softmax(h_fc3)

    return out_y


'''
LSTM
'''

lstm_time_step = 200
lstm_hidden_units = 64
lstm_layer_num = 1
lstm_input_dimension = 360

initializer = tf.contrib.layers.xavier_initializer()

lstm_weights = {

    'in': tf.Variable(initializer([lstm_input_dimension, lstm_hidden_units])),
    'out': tf.Variable(initializer([lstm_hidden_units, classes]))

}
lstm_biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[lstm_hidden_units, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[classes, ]))
}


def LSTM(x, weights, biases):
    x = tf.reshape(x, [-1, lstm_time_step, lstm_input_dimension])

    lstm_cell = rnn.BasicLSTMCell(num_units=lstm_hidden_units, forget_bias=1.0, state_is_tuple=True)

    lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=0.1)

    mlstm_cell = rnn.MultiRNNCell([lstm_cell] * lstm_layer_num, state_is_tuple=True)

    init_state = mlstm_cell.zero_state(train_batch, dtype=tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(mlstm_cell, inputs=x, initial_state=init_state, time_major=False)

    # outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    #
    # results = tf.matmul(outputs[-1], weights['out']) + biases['out']  # 选取最后一个 output

    return outputs


'''
the parameters of global
'''
global_classes = 5
train_path = '/home/dmrfcoder/Document/CSI/DataSet/new/fixed/traindatafixed.tfrecords'
val_path = '/home/dmrfcoder/Document/CSI/DataSet/new/fixed/testdatafixed.tfrecords'
pb_file_path = "../Model/CSI.pb"
log_path = "../Log/"
#
# x_train, y_train = read_and_decode(train_path)
# x_val, y_val = read_and_decode(val_path)

classes = 5
train_batch = 64
test_batch = 32
base_lr = 0.0001
global_training_iterations = 10  # 训练迭代次数
global_testing_iterations = 10  # 训练迭代次数
steps_per_test = 1  # train_batch * 2

lstm_input = tf.placeholder(tf.float32, shape=[None, lstm_time_step * lstm_input_dimension], name='input_lstm')
y_label = tf.placeholder(tf.int64, shape=[None, 1])

cnn_input = LSTM(lstm_input, lstm_weights, lstm_biases)
cnn_output = CNN(cnn_input)
#
# cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_label, logits=cnn_output)
#
# train = tf.train.GradientDescentOptimizer(learning_rate=base_lr).minimize(cross_entropy)
#
# argmax = tf.argmax(cnn_output, 1)
# correct_prediction = tf.equal(argmax, y_label)
#
# with tf.name_scope('loss'):
#     loss = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     tf.summary.scalar('loss', loss)

with tf.name_scope('train'):  # 训练和评估模型
    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=cnn_output))
        tf.summary.scalar('loss', cross_entropy)

    train = tf.train.GradientDescentOptimizer(learning_rate=base_lr).minimize(cross_entropy)
    argmax = tf.argmax(cnn_output, 1)
    correct_prediction = tf.equal(argmax, tf.argmax(y_label, 1))

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)


min_after_dequeue_train = train_batch * 2
min_after_dequeue_test = test_batch * 2

num_threads = 3  # 开启3个线程

train_capacity = min_after_dequeue_train + num_threads * train_batch
test_capacity = min_after_dequeue_test + num_threads * test_batch

'''
# 使用shuffle_batch可以随机打乱输入
train_x_batch, train_y_batch = tf.train.shuffle_batch([x_train, y_train],
                                                      batch_size=train_batch, capacity=train_capacity,
                                                      min_after_dequeue=min_after_dequeue_train)

# 使用shuffle_batch可以随机打乱输入
test_x_batch, test_y_batch = tf.train.shuffle_batch([x_val, y_val],
                                                    batch_size=test_batch, capacity=test_capacity,
                                                 min_after_dequeue=min_after_dequeue_test)
'''

train_f_p = open('../Data/train_pr_label.txt', 'wb')
train_f_r = open('../Data/train_re_label.txt', 'wb')

test_f_p = open('../Data/test_pr_label.txt', 'wb')
test_f_r = open('../Data/test_re_label.txt', 'wb')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    threads = tf.train.start_queue_runners(sess=sess)

    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_path, graph=sess.graph)

    for step in range(global_training_iterations + 1):
        # train_x, train_y = sess.run([train_x_batch, train_y_batch])
        # [batch, height, width, channels]
        train_x = np.random.rand(train_batch, 200 * 360)
        train_y = np.random.randint(0, 5, size=(train_batch, 1))
        sess.run(train,feed_dict={lstm_input: train_x, y_label: train_y})
        # # 写入日志
        # summary, _ = sess.run([merged, train], feed_dict={lstm_input: train_x, y_label: train_y})
        # writer.add_summary(summary, step)

        predict_labels = sess.run(argmax, feed_dict={lstm_input: train_x, y_label: train_y})

        np.savetxt(train_f_p, predict_labels)
        np.savetxt(train_f_r, train_y)

        if step % steps_per_test == 0:
            train_accuracy = sess.run(accuracy,feed_dict={lstm_input: train_x, y_label: train_y})

            print('Training Accuracy', step, train_accuracy)
            summary = sess.run(merged, feed_dict={lstm_input: train_x, y_label: train_y})
            summary_writer.add_summary(summary, step)


    train_f_p.close()
    train_f_r.close()
    summary_writer.close()

    for step in range(global_testing_iterations + 1):
        #test_x, test_y = sess.run([test_x_batch, test_y_batch])
        test_x = np.random.rand(train_batch, 200 * 360)
        test_y = np.random.randint(0, 5, size=(train_batch, 1))
        b = sess.run(accuracy, feed_dict={lstm_input: test_x, y_label: test_y})
        predict_labels = sess.run(argmax, feed_dict={lstm_input: train_x, y_label: train_y})
        np.savetxt(test_f_p, predict_labels)
        np.savetxt(test_f_r, test_y)
        print('Test Accuracy', step,b)
    test_f_p.close()
    test_f_r.close()

