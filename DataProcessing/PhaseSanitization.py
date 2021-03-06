# -*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt

'''
Input:
    raw phase :PM(180--->6*30)
    number of subcarriersn :Sub(30)
    number of Tx-Rx pairs:M(6)
    
Output:
    calibrated phase values:PC
    
Algorithm:
    1:unwrap
    2:polyfit
'''


def plotPhase(data, name, xlable='', ylable=''):
    plt.plot(data)

    plt.xlabel(xlable)
    plt.ylabel(ylable)

    plt.savefig('../EResult/' + name + '.png')
    plt.close()


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



