# -*_coding:utf-8-*-
import scipy.io as scio
import tensorflow as tf
import numpy as np


# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串类型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def DoConvert(path_mat, path_tfrecords,key_x,key_y):
    writer = tf.python_io.TFRecordWriter(path_tfrecords)
    mat_data = scio.loadmat(path_mat)
    x = mat_data[key_x]
    y = mat_data[key_y]
    item_count = y.shape[0]
    '''
    一共有item_count数量的数据,x的数据类型为float64,每一个x的维度为1*360
    已经归一化处理过
    '''

    for i in range(0, item_count):
        data_raw = x[i]
        label = np.where(y[i] == 1)[0][0]
        data_bytes=data_raw.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(label),
            'data_raw': _bytes_feature(data_bytes)
        }))
        print('doing:' + str(i)+' label:'+str(label))
        writer.write(example.SerializeToString())
    writer.close()
    print('success')



if __name__ == '__main__':
    path_mat='/home/dmrfcoder/Document/CSI/DataSet/new/fixed/traindatafixed.mat'
    path_tfrecords='/home/dmrfcoder/Document/CSI/DataSet/new/fixed/traindatafixed.tfrecords'
    key_x='train_x'
    key_y='train_y'
    DoConvert(path_mat,path_tfrecords,key_x,key_y)