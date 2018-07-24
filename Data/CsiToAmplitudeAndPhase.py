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

    """
    根据复数计算振幅和相位
    """
    amplitudes = [([] * 180) for i in range(N)]
    phases = [([] * 180) for i in range(N)]
    for m in range(N):
        for i in range(180):
            r = sqrt((csi_data[m][i].real) ** 2 + (csi_data[m][i].imag) ** 2)
            amplitudes[m].append(r)
            phases[m].append(np.angle(csi_data[m][i]))
    return (amplitudes, phases)


if __name__=='__main__':
    path='/home/dmrfcoder/Document/CSI/DataSet/new/fixed/eating/1/eating_1_2.mat'
    c=getAmplitudesAndPhases(path)
    print(0)