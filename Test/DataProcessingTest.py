# -*-coding:utf-8-*-

import scipy.io as scio

import DataProcessing


def DataProcessingTest():
    datapath = ''
    result = DataProcessing.DataProcessing.DataProcessing(datapath)
    amplitudes = result[0]
    phases = result[1]
    l = len(amplitudes)
    list = []
    for i in range(0, l):
        list.append(amplitudes[i] + phases[i])

    scio.savemat('demo.mat', {'key', list})
