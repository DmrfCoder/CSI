# -*-coding:utf-8-*-
from matplotlib.ticker import MultipleLocator

import numpy as np
import matplotlib.pyplot as plt

from DataProcessing.CsiToAmplitudeAndPhase import getAmplitudesAndPhases
from DataProcessing.PhaseSanitization import PhaseSanitization
from DataProcessing.WeightedMovingAverage import weightMoveAverage


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

    plotAmplitudes(amplitudes[:, 29, 0], 'Raw_data_after_pca')
    amplitudes = weightMoveAverage(amplitudes, N)
    plotAmplitudes(amplitudes[:, 29, 0], 'Weight_moving_average_filter')

    for k in range(0, N):
        phases[k] = PhaseSanitization(phases[k], 30, 6)

    return amplitudes, phases
    # amplitudes和amplitudes的维度都是n*180，现在将其写为n*360即可，但是要注意打上label


if __name__ == '__main__':
    path = '/Users/dmrfcoder/Documents/eating/1/eating_1_1.mat'
    result = DataProcessing(path)
