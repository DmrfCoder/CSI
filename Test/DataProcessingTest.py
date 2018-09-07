# -*-coding:utf-8-*-
import random

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


train_batch = 64
lens = 100
train_indexs = list(range(0, lens))
for i in range(0, 100):
    if len(train_indexs) < train_batch:
        train_indexs = list(range(0, lens))

    indexs = random.sample(range(0, len(train_indexs)), train_batch)

    sort = sorted(indexs, reverse=True)

    for ind in sort:
        print(ind, len(train_indexs))

        train_indexs.pop(ind)
