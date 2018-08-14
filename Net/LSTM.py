import tensorflow as tf

from DlTrain.TrainNet import classes, train_batch

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
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tenosors list of shape (batch_size, n_input)
    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.reshape(x, [-1, lstm_time_step, lstm_input_dimension])

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_hidden_units, forget_bias=1.0, state_is_tuple=True)

    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=0.1)

    # mlstm_cell = rnn.MultiRNNCell([lstm_cell] * lstm_layer_num, state_is_tuple=True)
    mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * lstm_layer_num, state_is_tuple=True)

    init_state = mlstm_cell.zero_state(train_batch, dtype=tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(mlstm_cell, inputs=x, initial_state=init_state, time_major=False)
    # outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))

    results = tf.matmul(outputs[-1], weights['out']) + biases['out']  # 选取最后一个 output

    return results
