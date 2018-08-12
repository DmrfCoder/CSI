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
    plotAmplitudes(amplitude[:, 29, 0], 'Raw_data_after_pca')
    m_item_sum = mItemSum(m)

    for a in range(0, 30):
        for t in range(0, 6):
            ass = amplitude[:, a, t]
            amplitude[:, a, t] = demo(ass, m, N, m_item_sum)

    plotAmplitudes(amplitude[:, 29, 0], 'Raw_data_after_pca2')
    print('success')
    return amplitude


def mItemSum(m):
    m_item_sum = 0
    for i in range(1, m + 1):
        m_item_sum = m_item_sum + i
    return m_item_sum


def demo(ass, m, N, m_item_sum):
    for n in range(m - 1, N):
        sum = 0
        x = ass[n - m + 1:n + 1]
        y = range(1, m + 1)
        for a, b in zip(x, y):
            sum += a * b
        ass[n] = sum / m_item_sum
    return ass


a = np.random.random(size=[2000, 30, 6])
b = weightMoveAverage(a, 2000)
