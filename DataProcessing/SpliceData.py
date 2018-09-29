# -*-coding:utf-8-*-
import os
import time

import scipy.io as scio
import numpy as np
import tensorflow as tf
import shutil


# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串类型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def spliceList(pathA, pathP, targetPath, files, folderName, fraction, start_time, fragmentLength=200):
    if not os.path.exists(targetPath + '/' + folderName):
        os.mkdir(targetPath + '/' + folderName)
    else:
        shutil.rmtree(targetPath + '/' + folderName)
        os.mkdir(targetPath + '/' + folderName)

    writer_train = tf.python_io.TFRecordWriter(targetPath + '/' + folderName + '/' + 'train.tfrecords')
    writer_val = tf.python_io.TFRecordWriter(targetPath + '/' + folderName + '/' + 'val.tfrecords')

    files_size = float(len(files))
    count = float(1)

    for file in files:
        data_A = scio.loadmat(pathA + '/' + file)
        data_P = scio.loadmat(pathP + '/' + file)

        data_A_x = data_A['x']
        data_A_y = data_A['y']
        data_P_x = data_P['x']

        x = np.concatenate((data_A_x, data_P_x), axis=1)

        length = data_A_x.shape[0]
        y = data_A_y[0][0]

        index = int(length / fragmentLength)

        trainindex = int(index * 0.8)

        traincount = 0
        valcount = 0

        for i in range(index):
            localx = np.array(x[i * fragmentLength:(i + 1) * fragmentLength])  # 200*360
            data_raw = localx.tostring()

            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(y),
                'data_raw': _bytes_feature(data_raw)
            }))

            if i <= trainindex:
                traincount += 1
                writer_train.write(example.SerializeToString())
            else:
                valcount += 1
                writer_val.write(example.SerializeToString())

        t_fraction=round(count / files_size,5)
        fraction = round(fraction + t_fraction/6, 4)
        now_time = time.time()
        secod = round(now_time - start_time, 2)
        fenzhong = int(secod / 60)
        secod = round(secod - fenzhong * 60, 2)
        xiaoshi = int(fenzhong / 60)
        fenzhong = fenzhong - xiaoshi * 60

        str1 = '已经进行：' + str(fraction) + '% ,用时：' + str(xiaoshi) + '小时' + str(fenzhong) + '分钟' + str(
            secod) + '秒 预计还需：'

        secod = round(secod / fraction, 2)
        fenzhong = int(secod / 60)
        secod = round(secod - fenzhong * 60, 2)
        xiaoshi = int(fenzhong / 60)
        fenzhong = fenzhong - xiaoshi * 60

        str1 = str1 + str(xiaoshi) + '小时' + str(fenzhong) + '分钟' + str(secod) + '秒'
        print str1

    writer_train.close()
    writer_val.close()


def spliceListSingle(pathA, targetPath, files, folderName, fraction, start_time, fragmentLength=200):
    if not os.path.exists(targetPath + '/' + folderName):
        os.mkdir(targetPath + '/' + folderName)
    else:
        shutil.rmtree(targetPath + '/' + folderName)
        os.mkdir(targetPath + '/' + folderName)

    writer_train = tf.python_io.TFRecordWriter(targetPath + '/' + folderName + '/' + 'train.tfrecords')
    writer_val = tf.python_io.TFRecordWriter(targetPath + '/' + folderName + '/' + 'val.tfrecords')

    files_size = float(len(files))
    count = float(1)

    for file in files:
        data_A = scio.loadmat(pathA + '/' + file)

        x = data_A['x']
        data_A_y = data_A['y']

        length = x.shape[0]
        y = data_A_y[0][0]

        index = int(length / fragmentLength)

        trainindex = int(index * 0.8)

        traincount = 0
        valcount = 0

        for i in range(index):
            localx = np.array(x[i * fragmentLength:(i + 1) * fragmentLength])  # 200*360
            data_raw = localx.tostring()

            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(y),
                'data_raw': _bytes_feature(data_raw)
            }))

            if i <= trainindex:
                traincount += 1
                writer_train.write(example.SerializeToString())
            else:
                valcount += 1
                writer_val.write(example.SerializeToString())

        fraction = round(fraction + float(count / files_size), 2)
        now_time = time.time()
        secod = round(now_time - start_time, 2)
        fenzhong = int(secod / 60)
        secod = round(secod - fenzhong * 60, 2)
        xiaoshi = int(fenzhong / 60)
        fenzhong = fenzhong - xiaoshi * 60

        str1 = '已经进行：' + str(fraction) + '% ,用时：' + str(xiaoshi) + '小时' + str(fenzhong) + '分钟' + str(
            secod) + '秒 预计还需：'

        secod = round(secod / fraction, 2)
        fenzhong = int(secod / 60)
        secod = round(secod - fenzhong * 60, 2)
        xiaoshi = int(fenzhong / 60)
        fenzhong = fenzhong - xiaoshi * 60

        str1 = str1 + str(xiaoshi) + '小时' + str(fenzhong) + '分钟' + str(secod) + '秒'
        print str1

    writer_train.close()
    writer_val.close()


def spliceProcessSingle(pathA, targetPath, fraction, start_time):
    filesA = os.listdir(pathA)

    files_open = []
    files_semi = []
    files_fixed = []

    for file in filesA:
        if 'open' in file:
            files_open.append(file)
        elif 'semi' in file:
            files_semi.append(file)
        else:
            files_fixed.append(file)

    spliceListSingle(pathA, targetPath, files_open, 'open', fraction + 0, start_time)
    spliceListSingle(pathA, targetPath, files_open, 'semi', fraction + 1 / 3, start_time)
    spliceListSingle(pathA, targetPath, files_open, 'fixed', fraction + 2 / 3, start_time)

    print 'success single:' + targetPath


def spliceProcess(pathA, pathP, targetPath, fraction, start_time):
    filesA = os.listdir(pathA)

    files_open = []
    files_semi = []
    files_fixed = []

    for file in filesA:

        if 'open' in file:
            files_open.append(file)
        elif 'semi' in file:
            files_semi.append(file)
        else:
            files_fixed.append(file)

    spliceList(pathA, pathP, targetPath, files_open, 'open', fraction + 0, start_time)
    spliceList(pathA, pathP, targetPath, files_semi, 'semi', fraction + 1 / 3, start_time)
    spliceList(pathA, pathP, targetPath, files_fixed, 'fixed', fraction + 2 / 3, start_time)

    print 'success:' + targetPath


if __name__ == '__main__':
    AmplitudeWithNoiseRemoval = '/media/xue/Data Storage/CSI/MatData/AmplitudeWithNoiseRemoval'
    AmplitudeWithOutNoiseRemoval = '/media/xue/Data Storage/CSI/MatData/AmplitudeWithOutNoiseRemoval'
    PhaseWithNoiseRemoval = '/media/xue/Data Storage/CSI/MatData/PhaseWithNoiseRemoval'
    PhaseWithOutNoiseRemoval = '/media/xue/Data Storage/CSI/MatData/PhaseWithOutNoiseRemoval'

    AmplitudeWithout_PhaseWith = '/media/xue/Data Storage/CSI/TfRecordsData/AmplitudeWithout_PhaseWith'
    AmplitudeWithOut_PhaseWithout = '/media/xue/Data Storage/CSI/TfRecordsData/AmplitudeWithOut_PhaseWithout'
    AmplitudeWith_PhaseWith = '/media/xue/Data Storage/CSI/TfRecordsData/AmplitudeWith_PhaseWith'
    AmplitudeWith_PhaseWithout = '/media/xue/Data Storage/CSI/TfRecordsData/AmplitudeWith_PhaseWithout'

    OnlyAmplitude = '/media/xue/Data Storage/CSI/TfRecordsData/OnlyAmplitude'
    OnlyPhase = '/media/xue/Data Storage/CSI/TfRecordsData/OnlyPhase'

    start_time = time.time()

    # spliceProcess(AmplitudeWithOutNoiseRemoval, PhaseWithNoiseRemoval, AmplitudeWithout_PhaseWith, 0, start_time)
    # spliceProcess(AmplitudeWithOutNoiseRemoval, PhaseWithOutNoiseRemoval, AmplitudeWithOut_PhaseWithout, 1 / 6,
    #               start_time)
    # spliceProcess(AmplitudeWithNoiseRemoval, PhaseWithNoiseRemoval, AmplitudeWith_PhaseWith, 2 / 6, start_time)
    # spliceProcess(AmplitudeWithNoiseRemoval, PhaseWithOutNoiseRemoval, AmplitudeWith_PhaseWithout, 3 / 6, start_time)


    spliceProcessSingle(AmplitudeWithNoiseRemoval, OnlyAmplitude, 4 / 6, start_time)
    spliceProcessSingle( PhaseWithNoiseRemoval, OnlyPhase, 5 / 6, start_time)
