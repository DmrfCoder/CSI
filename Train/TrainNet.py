# -*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from Net.CNN_Init import cnn_weight_variable, cnn_bias_variable, conv2d, max_pool_2x2
from Util.ReadAndDecodeUtil import read_and_decode

'''
the parameters of LSTM
'''

lstm_time_step = 200
lstm_hidden_units = 64
lstm_layer_num = 1
lstm_input_dimension = 360

'''
the parameters of global
'''
global_classes = 5
train_path = '/home/dmrfcoder/Document/CSI/DataSet/new/fixed/traindatafixed.tfrecords'
val_path = '/home/dmrfcoder/Document/CSI/DataSet/new/fixed/testdatafixed.tfrecords'
pb_file_path = "../Model/CSI.pb"

# 组合batch
train_batch = 64
test_batch = 32
global_training_iterations = 100000  # 训练迭代次数
steps_per_test = train_batch*2

x_train, y_train = read_and_decode(train_path)
x_val, y_val = read_and_decode(val_path)

# the input shap of LSTM  = (batch_size, timestep_size, input_size)
lstm_input = tf.placeholder(tf.float32, shape=[None, lstm_time_step * lstm_input_dimension], name='input_lstm')
y_label = tf.placeholder(tf.int64, shape=[None, ])

initializer = tf.contrib.layers.xavier_initializer()

lstm_weights = {

    'in': tf.Variable(initializer([lstm_input_dimension, lstm_hidden_units])),
    'out': tf.Variable(initializer([lstm_hidden_units, global_classes]))

}
lstm_biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[lstm_hidden_units, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[global_classes, ]))
}

'''
LSTM Net
'''


def LSTM(x, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tenosors list of shape (batch_size, n_input)
    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.reshape(x, [-1, lstm_time_step, lstm_input_dimension])

    #lstm_cell = rnn.BasicLSTMCell(num_units=lstm_hidden_units, forget_bias=1.0, state_is_tuple=True)
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_hidden_units, forget_bias=1.0, state_is_tuple=True)

    #mlstm_cell = rnn.MultiRNNCell([lstm_cell] * lstm_layer_num, state_is_tuple=True)
    mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * lstm_layer_num, state_is_tuple=True)

    init_state = mlstm_cell.zero_state(train_batch, dtype=tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(mlstm_cell, inputs=x, initial_state=init_state, time_major=False)
    #outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))

    results = tf.matmul(outputs[-1], weights['out']) + biases['out']  # 选取最后一个 output

    return results


'''
CNN Net
'''


def CNN(in_x):
    # [filter_height, filter_width, in_channels, out_channels]
    w_conv1 = cnn_weight_variable([5, 5, 1, 6])
    b_conv1 = cnn_bias_variable([6])
    try:
        h_conv1 = tf.nn.relu(conv2d(in_x, w_conv1, [1, 1, 1,1]) + b_conv1)  # stride/kernel:The stride of the sliding window for each  dimension of `input`.
    except  ValueError:
            print('error')
    h_pool1 = max_pool_2x2(h_conv1, [2, 2, 1, 1],
                           [2, 2, 1,
                            1])  # stride/kernel:The size of the window for each dimension of the input tensor.

    w_conv2 = cnn_weight_variable([5, 3, 6, 10])
    b_conv2 = cnn_bias_variable([10])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2, s=[3, 3, 1, 1]) + b_conv2)

    w_fc1 = cnn_weight_variable([3200, 1000])
    b_fc1 = cnn_bias_variable([1000])
    h_pool3_flat = tf.reshape(h_conv2, [-1, 3200])  # 将32*10*10reshape为3200*1

    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1) + b_fc1)

    w_fc2 = cnn_weight_variable([1000, 200])
    b_fc2 = cnn_bias_variable([200])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)

    w_fc3 = cnn_weight_variable([200, global_classes])
    b_fc3 = cnn_bias_variable([global_classes])
    h_fc3 = tf.nn.relu(tf.matmul(h_fc2, w_fc3) + b_fc3)

    out_y = tf.nn.softmax(h_fc3)

    return out_y


cnn_input = LSTM(lstm_input, lstm_weights, lstm_biases)
cnn_output = CNN(cnn_input)

prediction_labels = tf.argmax(y, axis=1, name="output")

cnn_base_lr = 0.1

cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_label, logits=cnn_output)

train = tf.train.GradientDescentOptimizer(learning_rate=cnn_base_lr).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(cnn_output, 1), y_label)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



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

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    threads = tf.train.start_queue_runners(sess=sess)
    for step in range(global_training_iterations + 1):
        train_x, train_y = sess.run([train_x_batch, train_y_batch])
        train_x = np.reshape(train_x, (1, 360, 1, 1))

        sess.run(train, feed_dict={lstm_input: train_x, y_label: train_y})
        if step % steps_per_test == 0:
            print('Training Accuracy', step,
                  sess.run(train, feed_dict={lstm_input: train_x, y_label: train_y}))

    constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])

    with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
        f.write(constant_graph.SerializeToString())
