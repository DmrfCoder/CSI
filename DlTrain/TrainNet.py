# -*-coding:utf-8-*-
import random

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from Util.loadHd import load

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

    out_y = tf.nn.softmax(h_fc3,name='cnn_softmax')

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


log_path = "../Log/"



pb_file_path = "../Data/CSI_open.pb"
ckpt_path='../ckpt_open/'

#f=load('F:\\csi\\openh5\\open_train.h5')
x=[]#f[0]
y=[]#f[1]
le=len(y)



f_val=load('F:\\csi\\openh5\\open_val.h5')
x_val=f_val[0]
y_val=f_val[1]
le_val=len(y_val)



train_batch = 16
val_batch = train_batch
global_training_iterations = 30000  # 训练迭代次数
global_val_iterations=10000
steps_per_test = 1  # train_batch * 2

lstm_input = tf.placeholder(tf.float32, shape=[None, lstm_time_step * lstm_input_dimension], name='input_lstm')
y_label = tf.placeholder(tf.int32, shape=[None, ])

cnn_input = LSTM(lstm_input, lstm_weights, lstm_biases)
cnn_output = CNN(cnn_input)


with tf.name_scope('base_lr'):
    base_lr =0.0001
    tf.summary.scalar('base_lr', base_lr) #写入tensorboard中的EVENTS

with tf.name_scope('loss'):
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y_label, logits=cnn_output)
    tf.summary.scalar('loss', loss)


train_op = tf.train.GradientDescentOptimizer(learning_rate=base_lr).minimize(loss)

pre_label=tf.cast(tf.argmax(cnn_output, 1),tf.int32)
correct_prediction = tf.equal(pre_label, y_label)
with tf.name_scope('acc'):
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('acc', acc)


'''
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

'''



train_f_p = open('../Data/train_pr_label.txt', 'wb')
train_f_r = open('../Data/train_re_label.txt', 'wb')

test_f_p = open('../Data/test_pr_label.txt', 'wb')
test_f_r = open('../Data/test_re_label.txt', 'wb')

def convert_y(labels):
    l_c=len(labels)
    y_expend=np.zeros(shape=(l_c))
    for lindex in range(0,l_c):
        y_expend[lindex]=labels[lindex]

    return y_expend


sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

is_train=True
saver=tf.train.Saver(max_to_keep=3)
merged = tf.summary.merge_all()

if is_train:
    writer = tf.summary.FileWriter(log_path, sess.graph)

    max_acc = 0
    train_indexs = list(range(0, le))
    epoch=1
    per_steps=int(le/train_batch)
    print('per_steps:'+str(per_steps))

    f = open(ckpt_path+'acc.txt', 'w')
    f_max = open(ckpt_path+'max_acc.txt', 'w')
    for step in range(global_training_iterations + 1):

        

        if len(train_indexs) < train_batch:
            train_indexs=list(range(0,le))
            epoch+=1
            break


        indexs =random.sample(range(0, len(train_indexs)), train_batch)
        #indexs =random.sample(range(0, le), train_batch)
        trainx = []
        trainy = []

        for index in indexs:
            trainx.append(x[index])
            trainy.append(y[index])

        sort_index=sorted(indexs,reverse = True)
        #如果还抛异常用可以手动捕捉一下跳过
        for index2 in sort_index:
            train_indexs.pop(index2)


        train_x=np.reshape(trainx,newshape=(-1,72000))
        train_y=np.reshape(trainy,newshape=(-1,1))
        if step==0:
            print(trainy)
        np.savetxt(train_f_r, train_y)
        train_y=convert_y(train_y)
        '''
        train_x = np.random.random(size=(train_batch, 72000))
        train_y = np.random.randint(0, 5, size=(train_batch))
        '''
        sess.run(train_op, feed_dict={lstm_input: train_x, y_label: train_y})

        train_loss, train_acc = sess.run([loss, acc], feed_dict={lstm_input: train_x, y_label: train_y})
        print('epoch:%d, step:%d,train_loss:%f, train_acc:%f' % (epoch,step ,train_loss, train_acc))
        f.write(str(step + 1) + ', train_acc: ' + str(train_acc) + '\n')
        if train_acc > max_acc:
            max_acc = train_acc
            f_max.write('max_acc: ' + str(max_acc) + '\n')

            saver.save(sess, ckpt_path+'csi.ckpt', global_step=step + 1)

        summary, _ = sess.run([merged, train_op], feed_dict={lstm_input: train_x, y_label: train_y})
        writer.add_summary(summary, step)
    f.close()
    f_max.close()
    writer.close()

else:
    model_file = tf.train.latest_checkpoint(ckpt_path)
    saver.restore(sess, model_file)
    val_indexs = list(range(0, le_val))

    for step in range(global_val_iterations):

        if len(val_indexs) < train_batch:
            val_indexs = list(range(0, le_val))
            break
        indexs = random.sample(range(0, len(val_indexs)), val_batch)
        valx = []
        valy = []
        for index in indexs:
            valx.append(x_val[index])
            valy.append(y_val[index])

        sort_val_index = sorted(indexs, reverse=True)
        for index2 in sort_val_index:
            val_indexs.pop(index2)

        val_x = np.reshape(valx, newshape=(-1, 72000))
        val_y = np.reshape(valy, newshape=(-1, 1))

        val_y = convert_y(val_y)
        np.savetxt(test_f_r, val_y)

        '''
        val_x = np.random.random(size=(val_batch, 72000))
        val_y = np.random.randint(0, 5, size=(val_batch))
        '''
        val_loss, val_acc = sess.run([loss, acc], feed_dict={lstm_input: val_x, y_label: val_y})
        print('val_loss:%f, val_acc:%f' % (val_loss, val_acc))


sess.close()