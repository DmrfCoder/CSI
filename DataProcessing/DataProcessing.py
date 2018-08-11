# -*-coding:utf-8-*-
import os
from math import sqrt

from matplotlib.ticker import MultipleLocator

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import tensorflow as tf


def getAmplitudesAndPhases(Csi_Mat_Path):  # N为对应180的数量

    data = scio.loadmat(Csi_Mat_Path)
    csi_data = data['csi']
    N = len(csi_data)

    """
    根据复数计算振幅和相位
    """
    amplitudes = np.ndarray(shape=(N, 30, 6))
    phases = np.ndarray(shape=(N, 30, 6))

    for m in range(N):
        for i in range(0, 6):
            for j in range(0, 30):
                index = j + i * 30
                amplitudes[m][j][i] = sqrt(csi_data[m][index].real ** 2 + csi_data[m][index].imag ** 2)
                phases[m][j][i] = np.angle(csi_data[m][index])

    return amplitudes, phases, N


def weightMoveAverage(amplitude, N, m=100):  # N为Amplitude的长度
    m_item_sum = mItemSum(m)

    for a in range(0, 30):
        for t in range(0, 6):
            for n in range(m - 1, N):
                print(str(a) + ' ' + str(t) + ' ' + str(n))
                sum_amplitude = 0
                jUpIndex = n
                jLowIndex = n - m + 1
                for j in range(jLowIndex, jUpIndex):  # t-m+1~t,因为range右边为开,所以+1
                    sum_amplitude = sum_amplitude + amplitude[j][a][t] * (j - jLowIndex + 1)

                amplitude[n][a][t] = sum_amplitude / m_item_sum

    return amplitude


def mItemSum(m):
    m_item_sum = 0
    for i in range(1, m + 1):
        m_item_sum = m_item_sum + i
    return m_item_sum


def PhaseSanitization(pm, sub=30, m=6):
    for i in range(0, m):
        pm[:, i] = np.unwrap(pm[:, i])

    y = np.mean(pm, 1)
    pc = np.ndarray(shape=(30, 6))

    x = range(0, sub)
    p = np.polyfit(x, y, 1)
    yf = [p[0] * tx for tx in x]

    for t in range(0, m):
        for s in range(0, 30):
            pc[s][t] = pm[s][t] - yf[s]

    return pc


def plotPhase(data, name, xlable='', ylable=''):
    plt.plot(data)

    plt.xlabel(xlable)
    plt.ylabel(ylable)

    plt.savefig('../EResult/' + name + '.png')
    plt.close()


def plotAmplitudes(data, name, xlable='', ylable=''):
    plt.plot(data)

    plt.xlabel(xlable)
    plt.ylabel(ylable)

    plt.savefig('../EResult/' + name + '.png')
    plt.close()


def DataProcessing(Csi_Mat_Path):
    # 从原始csi数据中的复数计算的到振幅和相位
    amplitudes_and_phases = getAmplitudesAndPhases(Csi_Mat_Path)
    N = amplitudes_and_phases[2]

    amplitudes = amplitudes_and_phases[0]
    phases = amplitudes_and_phases[1]

    amplitudes = weightMoveAverage(amplitudes, N)

    for k in range(0, N):
        print(str(k / N) + '')
        phases[k] = PhaseSanitization(phases[k], 30, 6)

    a = amplitudes.reshape(N, 360)
    p = phases.reshape(N, 360)

    return a, p, N

    # amplitudes和amplitudes的维度都是n*180，现在将其写为n*360即可，但是要注意打上label


# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串类型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def dataProcessOfFiles(path, targetpath):  # path:.../new targetpath:../demo.tfrecords

    writer = tf.python_io.TFRecordWriter(targetpath)

    files1 = os.listdir(path)

    a = -1
    b = -1
    c = -1
    d = -1

    for f1 in files1:
        a = a + 1
        # f1:fixed open semi
        files2 = os.listdir(path + '/' + f1)
        for f2 in files2:
            b = b + 1
            # f2:eatting settig ...
            files3 = os.listdir(path + '/' + f1 + '/' + f2)
            for f3 in files3:
                c = c + 1
                # f3:1 2 3...
                l = int(f3) - 1
                files4 = os.listdir(path + '/' + f1 + '/' + f2 + '/' + f3)
                for f4 in files4:
                    d = d + 1
                    print str(a) + ' ' + str(b) + ' ' + str(c) + ' ' + str(d)
                    # f4:1_1.mat 2_2.mat...
                    result = DataProcessing(path + '/' + f1 + '/' + f2 + '/' + f3 + '/' + f4)
                    a = result[0]
                    p = result[1]
                    N = result[2]
                    label = l
                    for n in range(0, N):
                        data_raw = a[n] + p[n]
                        data_bytes = data_raw.tostring()
                        example = tf.train.Example(features=tf.train.Features(feature={
                            'label': _int64_feature(label),
                            'data_raw': _bytes_feature(data_bytes)
                        }))

                        writer.write(example.SerializeToString())

    writer.close()
    print('success')


if __name__ == '__main__':
    path = '../new'
    tf_path = '../Data/csi_data.tfrecords'
