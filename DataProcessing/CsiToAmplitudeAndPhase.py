# -*-coding:utf-8-*-
from math import *
import numpy as np
import scipy.io as scio

'''
 Nrx: 3 接收端天线数量
 Ntx: 2 发送端天线数量
 rate:csi[rate] 不同数据包的rate不一样
 csi: [2x3x30 double] 180
 
'''



def getAmplitudesAndPhases(Csi_Mat_Path):  # N为对应180的数量

    data = scio.loadmat(Csi_Mat_Path)
    csi_data = data['csi']
    N = len(csi_data)

    '''
    设置N为10方便调试
    '''


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

    return amplitudes, phases,N
