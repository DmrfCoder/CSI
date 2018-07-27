# -*-coding:utf-8-*-
import numpy as np

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


def PhaseSanitization(PM, Sub, M):
    for i in range(0, M):
        Up = np.unwrap(PM[:][i])

    y = np.mean(PM, 2)
    PC = np.ndarray(shape=(6, 30))

    for i in range(0, Sub):
        x = range(0, Sub - 1)
        p = np.polyfit(x, y, 1)
        yf = p[0] * x
        for j in range(0, M):
            PC[:][i] = PM[:][i] - yf

    return PC
