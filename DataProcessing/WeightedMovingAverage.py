# -*-coding:utf-8-*-

'''
对Amplitude使用加权移动平均法进行去噪处理(平滑处理)
设置m为100

用法:Amplitude=WeightMoveAverage(Amplitude, N, m=100)

'''

import numpy as np
import matplotlib.pyplot as plt


def plotAmplitudes(data, name, xlable='', ylable=''):
    plt.plot(data)

    plt.xlabel(xlable)
    plt.ylabel(ylable)

    plt.savefig('../EResult/' + name + '.png')
    plt.close()


def weightMoveAverage(amplitude, N, m=100):  # N为Amplitude的长度
    m_item_sum = mItemSum(m)
    plotAmplitudes(amplitude[:, 29, 0], 'Raw_data_after_pca')

    for a in range(0, 30):
        for t in range(0, 6):
            for n in range(m - 1, N):
                sum_amplitude = 0
                jUpIndex = n
                jLowIndex = n - m + 1
                for j in range(jLowIndex, jUpIndex):  # t-m+1~t,因为range右边为开,所以+1
                    sum_amplitude = sum_amplitude + amplitude[j][a][t] * (j - jLowIndex + 1)

                amplitude[n][a][t] = sum_amplitude / m_item_sum

    plotAmplitudes(amplitude[:, 29, 0], 'Raw_data_after_pca2')
    return amplitude


def mItemSum(m):
    m_item_sum = 0
    for i in range(1, m + 1):
        m_item_sum = m_item_sum + i
    return m_item_sum


a = np.random.random((1000, 30, 6))

weightMoveAverage(a, 1000)
