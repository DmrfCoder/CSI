# -*-coding:utf-8-*-
# 读取文件。
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(
<<<<<<< HEAD
    ['E:\\yczhao Data\\open.tfrecords'])
=======
    ['demo.tfrecords'])
>>>>>>> ca38e2faf428cfaa66bb64e10c3d32227dae4e03
_, serialized_example = reader.read(filename_queue)

# 解析读取的样例。
features = tf.parse_single_example(
    serialized_example,
    features={
        'label': tf.FixedLenFeature([], tf.int64),
        'data_raw': tf.FixedLenFeature([], tf.string)
    })

images = tf.decode_raw(features['data_raw'], tf.float64)
#images = tf.reshape(images, [1, 360])

labels = tf.cast(features['label'], tf.int64)



sess = tf.Session()

# 启动多线程处理输入数据。
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for i in range(20):
    image, label = sess.run([images, labels])
    print(label)



<<<<<<< HEAD
   # print label
=======
    print (image,label,i)
>>>>>>> ca38e2faf428cfaa66bb64e10c3d32227dae4e03
