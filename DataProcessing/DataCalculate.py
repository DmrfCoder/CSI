
# -*-coding:utf-8-*-

from DataProcessing.CsiToAmplitudeAndPhase import getAmplitudesAndPhases
from DataProcessing.PhaseSanitization import PhaseSanitization
from DataProcessing.WeightedMovingAverage import weightMoveAverage


def DataCalculate(Csi_Mat_Path):
    # 从原始csi数据中的复数计算的到振幅和相位
    amplitudes_and_phases = getAmplitudesAndPhases(Csi_Mat_Path)
    N = amplitudes_and_phases[2]

    amplitudes = amplitudes_and_phases[0]
    phases = amplitudes_and_phases[1]
    phases2=phases

    amplitudes2 = weightMoveAverage(amplitudes, N)

    for k in range(0, N):
        phases2[k] = PhaseSanitization(phases[k], 30, 6)
        break

    return amplitudes,phases, amplitudes2, phases2,N
    # amplitudes和amplitudes的维度都是n*180，现在将其写为n*360即可，但是要注意打上label


#DataCalculate('/media/xue/软件/CSI/RawMatData/fixed/eating/1/eating_1_1.mat')
