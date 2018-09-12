# -*-coding:utf-8-*-
import tensorflow as tf


# 读取tfrecords数据
def read_and_decode(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

<<<<<<< HEAD
=======

>>>>>>> c7c16b06acb9e61b60dd9bfbe34bf7628c81935b
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'data_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['data_raw'], tf.float64)
    img = tf.reshape(img, [200, 360])
    label = tf.cast(features['label'], tf.int64)

    return img, label


<<<<<<< HEAD
def test():
    train_path = '/data/after-split200/semi/train.tfrecords'

    x_train, y_train = read_and_decode(train_path)

    num_threads = 3

    min_after_dequeue_train = 10000
    train_batch = 64
    train_capacity = min_after_dequeue_train + num_threads * train_batch

    # 使用shuffle_batch可以随机打乱输入
    train_x_batch, train_y_batch = tf.train.shuffle_batch([x_train, y_train],
                                                          batch_size=train_batch, capacity=train_capacity,
                                                          min_after_dequeue=min_after_dequeue_train)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        threads = tf.train.start_queue_runners(sess=sess)

        for step in range(100):
            train_x, train_y = sess.run([train_x_batch, train_y_batch])
            print train_y

=======
train_path = 'F:\\csi\\open.tfrecords'
val_path = 'F:\\csi\\open_val.tfrecords'

x_train, y_train = read_and_decode(train_path)
x_val, y_val = read_and_decode(val_path)


# 使用shuffle_batch可以随机打乱输入
train_x_batch, train_y_batch = tf.train.batch([x_train, y_train],
                                                      batch_size=64)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    threads = tf.train.start_queue_runners(sess=sess)

    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_path, graph=sess.graph)

    for step in range(global_training_iterations + 1):
        train_x, train_y = sess.run([train_x_batch, train_y_batch])
        train_x=np.reshape(train_x,newshape=(-1,72000))
        train_y=np.reshape(train_y,newshape=(-1,1))
>>>>>>> c7c16b06acb9e61b60dd9bfbe34bf7628c81935b

