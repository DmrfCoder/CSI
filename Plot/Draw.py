# -*-coding:utf-8-*-
from matplotlib import pyplot as plt
import scipy.io as scio
import numpy as np

def Draw(path):
    data = scio.loadmat(path)
    Amplitude = data['X_normAllAmplitude'][0]
    Phase = data['X_normAllPhase'][0]

    plt.subplot(2, 1, 1)
    my_x_ticks = np.arange(0, 180, 60)
    plt.xticks(my_x_ticks)
    plt.title('X_normAllAmplitude')
    plt.plot(Amplitude)
    plt.subplot(2, 1, 2)
    plt.xticks(my_x_ticks)
    plt.plot(Phase)
    plt.title('X_normAllPhase')
    plt.show()


if __name__ == '__main__':
    path = '/home/dmrfcoder/Document/CSI/DataSet/new/fixed/eating/1/eating_1.mat'
    Draw(path)
