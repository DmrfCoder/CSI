# -*-coding:utf-8-*-
import tensorflow as tf


# 读取tfrecords数据
def read_and_decode(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'data_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['data_raw'], tf.float64)
    img = tf.reshape(img, [1, 360])
    label = tf.cast(features['label'], tf.int64)

    return img, label