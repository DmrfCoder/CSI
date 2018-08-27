# -*-coding:utf-8-*-
import os

from threading import Thread
import scipy.io as scio

from matplotlib.ticker import MultipleLocator

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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

    amplitudes = weightMoveAverage(amplitudes, N)

    for k in range(0, N):
        phases[k] = PhaseSanitization(phases[k], 30, 6)
        break

    return amplitudes, phases, N
    # amplitudes和amplitudes的维度都是n*180，现在将其写为n*360即可，但是要注意打上label


# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串类型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

t=1

def dataProcessOfFixedFiles(path, targetpath):  # path:.../new targetpath:../demo.tfrecords

#    writer = tf.python_io.TFRecordWriter(targetpath)

    files1 = os.listdir(path)

    a = -1
    b = -1
    c = -1
    d = -1


    files2 = os.listdir(path)
    for f2 in files2:
        b = b + 1
        # f2:eatting settig ...
        files3 = os.listdir(path +  '\\' + f2)
        for f3 in files3:
            c = c + 1
            # f3:1 2 3...
            l = int(f3) - 1
            files4 = os.listdir(path +'\\'+ f2 + '\\' + f3)
            for f4 in files4:
                if '.mat' in f4:
                    d = d + 1
                    print('dataProcessOfFixedFiles'+str(a) + ' ' + str(b) + ' ' + str(c) + ' ' + str(d))
                    # f4:1_1.mat 2_2.mat...
                    result = DataProcessing(path +  '\\' + f2 + '\\' + f3 + '\\' + f4)

                    ap = result[0]
                    p = result[1]
                    N = result[2]
                    label = [l]*N

                    scio.savemat(targetpath+ f2 + '-' + f3 + '-' + f4,
                                 {'x': np.concatenate((ap.reshape(N,-1),p.reshape(N,-1)),axis=1), 'y':label})

    print('success:'+path)



def dataProcessOfOpenAndSemiFiles(path, targetpath):  # path:.../new targetpath:../demo.tfrecords

    #writer = tf.python_io.TFRecordWriter(targetpath)


    a = -1
    b = -1
    c = -1
    d = -1


    files3 = os.listdir(path)

    for f3 in files3:
        c = c + 1
        # f3:1 2 3...
        l = int(f3) - 1
        files4 = os.listdir(path +  '\\' + f3)
        for f4 in files4:
            if '.mat' in f4:
                d = d + 1
                print('dataProcessOfOpenAndSemiFiles'+path+' '+str(a) + ' ' + str(b) + ' ' + str(c) + ' ' + str(d))
                # f4:1_1.mat 2_2.mat...
                result = DataProcessing(path+ '\\' + f3 + '\\' + f4)
                ap = result[0]
                p = result[1]
                N = result[2]
                label = [l]*N

                scio.savemat(targetpath  + f3 + '-' + f4,
                             {'x': np.concatenate((ap.reshape(N, -1), p.reshape(N, -1)), axis=1), 'y': label})


    print('success:'+path)




if __name__ == '__main__':
    path = '../new'
    tf_path = '../Data/csi_data.tfrecords'
    mat_path = '/Users/dmrfcoder/Documents/eating/1/eating_1_1.mat'



    fix_path='E:\\yczhao Data\\new\\fixed'
    fix_target_path='E:\\yczhao Data\\new-2-mat\\'
    #dataProcessOfFixedFiles(fix_path,fix_target_path)


    open_path='E:\\yczhao Data\\new\\open'
    open_target_path = 'E:\\yczhao Data\\new-2-mat1\\'
    #dataProcessOfOpenAndSemiFiles(open_path,open_target_path)

    semi_path='E:\\yczhao Data\\new\\semi'
    semi_target_path='E:\\yczhao Data\\new-2-mat2\\'
    #dataProcessOfOpenAndSemiFiles(semi_path,semi_target_path)

    threadsPool = [
                   Thread(target=dataProcessOfFixedFiles, args=(fix_path,fix_target_path)),
                  # Thread(target=dataProcessOfOpenAndSemiFiles, args=(open_path, open_target_path)),
                   # Thread(target=dataProcessOfOpenAndSemiFiles,args=(semi_path,semi_target_path))
                   ]
    for thread in threadsPool:
        thread.start()

    for thread in threadsPool:
        thread.join()





