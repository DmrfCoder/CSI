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
    y = range(1, m + 1)

    for a in range(0, 30):
        for t in range(0, 6):
            for n in range(m - 1, N):
                x = amplitude[n - m + 1:n + 1][a][t]
                sum_list = sum(map(lambda (a, b): a * b, zip(x, y)))
                amplitude[n][a][t] = sum_list / m_item_sum

    plotAmplitudes(amplitude[:, 29, 0], 'Raw_data_after_pca2')
    print 'sccess'
    return amplitude


def mItemSum(m):
    m_item_sum = 0
    for i in range(1, m + 1):
        m_item_sum = m_item_sum + i
    return m_item_sum


def demo(ass,m,N):
    y = range(1, m + 1)
    m_item_sum = mItemSum(m)
    for n in range(m - 1, N):
        x = ass[n - m + 1:n + 1]
        sum_list = sum(map(lambda (a, b): a * b, zip(x, y)))
        ass[n] = sum_list / m_item_sum
    return ass


a = np.random.random(2000)
plotAmplitudes(a,'a')
b=demo(a,100,2000)
plotAmplitudes(b,'b')