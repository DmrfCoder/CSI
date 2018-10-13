import tensorflow as tf
from tensorflow.contrib import rnn

from DlTrain.Parameters import lstmTimeStep, lstmHiddenUnits, lstmLayerNum, trainBatchSize

initializer = tf.contrib.layers.xavier_initializer()


def LSTM(x,lstmInputDimension):
    x = tf.reshape(x, [-1, lstmTimeStep, lstmInputDimension])

    lstm_cell = rnn.BasicLSTMCell(num_units=lstmHiddenUnits, forget_bias=1.0, state_is_tuple=True)

    lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=0.1)

    mlstm_cell = rnn.MultiRNNCell([lstm_cell] * lstmLayerNum, state_is_tuple=True)

    init_state = mlstm_cell.zero_state(trainBatchSize, dtype=tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(mlstm_cell, inputs=x, initial_state=init_state, time_major=False)

    return outputs
