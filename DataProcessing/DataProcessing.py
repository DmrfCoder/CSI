# -*-coding:utf-8-*-
import CsiToAmplitudeAndPhase
import WeightedMovingAverage
import PhaseSanitization
import numpy as np
import matplotlib.pyplot as plt


def DataProcessing(Csi_Mat_Path):
    AmplitudesAndPhases = CsiToAmplitudeAndPhase.getAmplitudesAndPhases(Csi_Mat_Path)

    amplitudes = AmplitudesAndPhases[0]  # n*180
    phases = np.array(AmplitudesAndPhases[1])  # n*180

    am = np.array(amplitudes[0]).reshape(6, 30)
    l1, = plt.plot(am[0], color='green', label='Antenna1')
    l2, = plt.plot(am[1], color='red', label='Antenna2')
    l3, = plt.plot(am[2], color='skyblue', label='Antenna3')

    plt.xlabel('Time')
    plt.ylabel('Amplitudes')

    plt.savefig('../EResult/Raw_data_after_PAC.png')
    plt.close()

    amplitudes = WeightedMovingAverage.WeightMoveAverage(amplitudes, N=len(amplitudes))

    a = np.array(amplitudes[0]).reshape(6, 30)
    plt.plot(a[0], color='green', label='Antenna1')
    plt.plot(a[1], color='red', label='Antenna2')
    plt.plot(a[2], color='skyblue', label='Antenna3')

    plt.xlabel('Time')
    plt.ylabel('Amplitudes')

    plt.savefig('../EResult/Weight_moving_average_filter.png')
    plt.close()

    a = np.array(phases[0]).reshape(6, 30)
    plt.plot(a[0], color='green', label='Antenna1')
    plt.plot(a[1], color='red', label='Antenna2')
    plt.plot(a[2], color='skyblue', label='Antenna3')

    plt.xlabel('Subcarrier')
    plt.ylabel('Raw Phase')

    plt.savefig('../EResult/Raw_wrapped_csi_phase.png')
    plt.close()

    for i in range(0, len(phases)):
        phases[i] = PhaseSanitization.PhaseSanitization(phases[i].reshape(6, 30), 30, 6).reshape(180)

    a = np.array(phases[0]).reshape(6, 30)
    plt.plot(a[0], color='green', label='Antenna1')
    plt.plot(a[1], color='red', label='Antenna2')
    plt.plot(a[2], color='skyblue', label='Antenna3')

    plt.xlabel('Subcarrier')
    plt.ylabel('Sanitise Phase')

    plt.savefig('../EResult/Modified_csi_phase.png')
    plt.close()

    return (amplitudes, phases)
    # amplitudes和amplitudes的维度都是n*180，现在将其写为n*360即可，但是要注意打上label


if __name__ == '__main__':
    path = '/Users/dmrfcoder/Documents/eating/1/eating_1_1.mat'
    result = DataProcessing(path)
