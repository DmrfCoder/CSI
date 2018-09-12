<<<<<<< HEAD
# coding:utf-8
=======
>>>>>>> c7c16b06acb9e61b60dd9bfbe34bf7628c81935b
import os

import scipy.io as scio

<<<<<<< HEAD
import tensorflow as tf

from Util.WriteHd5Util import writeToH5

import numpy as np
=======
from Util.WriteHd5Util import writeToH5
from Util.loadHd import bean
>>>>>>> c7c16b06acb9e61b60dd9bfbe34bf7628c81935b

import random


<<<<<<< HEAD
class bean:

    def __init__(self, x, y):
        self.x = x
        self.y = y


# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串类型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


'''
传入的数据维度为N*360&&360*1
将其进行切片后返回
'''


def split(x, y, fragmentLength=200):
    lenght = len(y[0])
    index = int(lenght / fragmentLength)

    tempx = []
    tempy = []

    for i in range(index):
        localx = x[i * fragmentLength:(i + 1) * fragmentLength]  # 200*360
        # localx = np.reshape(localx, newshape=(-1, fragmentLength * 360))
        tempx.append(localx)
        tempy.append(y[0][i])

    return (tempx, tempy)


def getshuffefilelist(files):
    fileslist = []

    for i in range(1, 6):
        for file in files:
            if file[0] == str(i):
                fileslist.append(file)
                files.remove(file)
                break

    return fileslist, files


'''
将文件夹里的mat文件逐个读出，然后对其进行split（切片），将其暂存在内存中，最后对其进行随机化并存储在h5文件中
'''


def SplitProcess(sourcePath, targetPath, fragmentLength=200):
    files = os.listdir(sourcePath)
    writer_train = tf.python_io.TFRecordWriter(targetPath + '/' + 'train.tfrecords')
    writer_val = tf.python_io.TFRecordWriter(targetPath + '/' + 'val.tfrecords')

    for file in files:
        print file
        data = scio.loadmat(sourcePath + '/' + file)
        tempx = data['x']
        tempy = data['y']

        lenght = len(tempy[0])
        index = int(lenght / fragmentLength)

        trainindex = int(index * 0.8)

        for i in range(index):
            localx = np.array(tempx[i * fragmentLength:(i + 1) * fragmentLength])  # 200*360
            data_raw = localx.tostring()
            label = int(tempy[0][i])
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(label),
                'data_raw': _bytes_feature(data_raw)
            }))

            if i <= trainindex:
                writer_train.write(example.SerializeToString())
            else:
                writer_val.write(example.SerializeToString())

    writer_train.close()
    writer_val.close()


if __name__ == '__main__':
    fixed_path = '/data/after-dataprocess/fixed/'
    fixed_hd_target = '/data/after-split200/fixed'

    SplitProcess(fixed_path, fixed_hd_target, 200)
=======
def split(sourcePath, targetPath, fragmentLength=200):
    files = os.listdir(sourcePath)
    x = []
    y = []
    for f1 in files:
        data = scio.loadmat(sourcePath + '\\' + f1)
        tempx = data['x']
        tempy = data['y']

        x.append(tempx)
        y.append(tempy)

    data = []
    l = len(y)
    for i in range(l):
        b = bean(x[i], y[i])
        data.append(b)

    random.shuffle(data)

    writeToH5(data, targetPath)


if __name__ == '__main__':
    fixed_path = 'F:\\csi\\AfterNormalizeData\\fixed\\'
    fixed_hd_target = 'F:\\csi\\AfterSplit200\\fixed.h5'

    split(fixed_path, fixed_hd_target, 200)
>>>>>>> c7c16b06acb9e61b60dd9bfbe34bf7628c81935b
