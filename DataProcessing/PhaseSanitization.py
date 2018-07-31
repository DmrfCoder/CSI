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


def PhaseSanitization(PM, Sub, M):
    for i in range(0, M):
        Up = np.unwrap(PM[i,:])

    plt.plot(PM[0,:], color='green', label='Antenna1')
    plt.plot(PM[1,:], color='red', label='Antenna2')
    plt.plot(PM[2,:], color='skyblue', label='Antenna3')



    plt.xlabel('Subcarrier')
    plt.ylabel('Unwrapped Phase')

    plt.savefig('../EResult/Unwrapped_csi_phase.png')
    plt.close()




    y = np.mean(PM, 0)
    PC = np.ndarray(shape=(6, 30))

    for i in range(0, Sub):
        x = range(0, Sub)
        p = np.polyfit(x, y, 1)
        yf = [p[0]*tx for tx in x]
        for j in range(0, M):
            PC[j, :]= list(map(lambda x: x[0] - x[1], zip(PM[j,:],  yf)))

    return PC


