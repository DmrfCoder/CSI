# -*-coding:utf-8-*-
import tensorflow as tf
import numpy as np

from Net.CNN import CNN
from Net.LSTM import LSTM, lstm_weights, lstm_biases, lstm_time_step, lstm_input_dimension
from Util.ReadAndDecodeUtil import read_and_decode

'''
the parameters of global
'''
global_classes = 5
train_path = '/home/dmrfcoder/Document/CSI/DataSet/new/fixed/traindatafixed.tfrecords'
val_path = '/home/dmrfcoder/Document/CSI/DataSet/new/fixed/testdatafixed.tfrecords'
pb_file_path = "../Model/CSI.pb"

x_train, y_train = read_and_decode(train_path)
x_val, y_val = read_and_decode(val_path)

classes = 5
train_batch = 64
test_batch = 32
base_lr = 0.0001
global_training_iterations = 100000  # 训练迭代次数
global_testing_iterations = 10000  # 训练迭代次数
steps_per_test = train_batch * 2

lstm_input = tf.placeholder(tf.float32, shape=[None, lstm_time_step * lstm_input_dimension], name='input_lstm')
y_label = tf.placeholder(tf.int64, shape=[None, ])

cnn_input = LSTM(lstm_input, lstm_weights, lstm_biases)
cnn_output = CNN(cnn_input)

cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_label, logits=cnn_output)

train = tf.train.GradientDescentOptimizer(learning_rate=base_lr).minimize(cross_entropy)

argmax = tf.argmax(cnn_output, 1)
correct_prediction = tf.equal(argmax, y_label)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('loss', loss)

min_after_dequeue_train = train_batch * 2
min_after_dequeue_test = test_batch * 2

num_threads = 3  # 开启3个线程

train_capacity = min_after_dequeue_train + num_threads * train_batch
test_capacity = min_after_dequeue_test + num_threads * test_batch

# 使用shuffle_batch可以随机打乱输入
train_x_batch, train_y_batch = tf.train.shuffle_batch([x_train, y_train],
                                                      batch_size=train_batch, capacity=train_capacity,
                                                      min_after_dequeue=min_after_dequeue_train)

# 使用shuffle_batch可以随机打乱输入
test_x_batch, test_y_batch = tf.train.shuffle_batch([x_val, y_val],
                                                    batch_size=test_batch, capacity=test_capacity,
                                                    min_after_dequeue=min_after_dequeue_test)

re_label_train = np.ndarray(6400, dtype=np.int64)
pr_label_train = np.ndarray(6400, dtype=np.int64)

train_f_p = open('../Data/train_pr_label.txt', 'wb')
train_f_r = open('../Data/train_re_label.txt', 'wb')

s_train = 0

re_label_test = np.ndarray(6400, dtype=np.int64)
pr_label_test = np.ndarray(6400, dtype=np.int64)

test_f_p = open('../Data/test_pr_label.txt', 'wb')
test_f_r = open('../Data/test_re_label.txt', 'wb')

s_test = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    threads = tf.train.start_queue_runners(sess=sess)
    for step in range(global_training_iterations + 1):
        train_x, train_y = sess.run([train_x_batch, train_y_batch])
        train_x = np.reshape(train_x, (1, 360, 1, 1))

        sess.run(train, feed_dict={lstm_input: train_x, y_label: train_y})

        predict_labels = sess.run(argmax)
        re_label_train[s_train:(s_train + 1) * train_batch] = train_y
        pr_label_train[s_train:(s_train + 1) * train_batch] = predict_labels

        if s_train == 10:
            np.savetxt(train_f_p, pr_label_train)
            np.savetxt(train_f_r, re_label_train)
            s_train += 1
        else:
            s_train = 0

        if step % steps_per_test == 0:
            print('Training Accuracy', step,
                  sess.run(train, feed_dict={lstm_input: train_x, y_label: train_y}))

    train_f_p.close()
    train_f_r.close()

    for step in range(global_testing_iterations + 1):
        test_x, test_y = sess.run([test_x_batch, test_y_batch])
        test_x = np.reshape(test_x, (1, 360, 1, 1))

        predict_labels = sess.run(argmax)
        re_label_test[s_test:(s_test + 1) * test_batch] = test_y
        pr_label_test[s_test:(s_test + 1) * test_batch] = predict_labels

        if s_test == 10:
            np.savetxt(test_f_p, pr_label_test)
            np.savetxt(test_f_r, re_label_test)
            s_test += 1
        else:
            s_test = 0

        print('Testing Accuracy', step,
              sess.run(train, feed_dict={lstm_input: test_x, y_label: test_y}))

    test_f_r.close()
    test_f_p.close()

    constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])

    with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
        f.write(constant_graph.SerializeToString())
