# -*-coding:utf-8-*-
import os

from threading import Thread
import scipy.io as scio

import numpy as np

from DataProcessing.DataCalculate import DataCalculate
from DataProcessing.Normalize import Normalize


def dataProcessOfFixedFiles(path, targetpath):  # path:.../new targetpath:../demo.tfrecords

    files2 = os.listdir(path)

    for f2 in files2:
        # f2:eatting settig ...
        files3 = os.listdir(path + '\\' + f2)
        for f3 in files3:
            # f3:1 2 3...
            l = int(f3) - 1
            files4 = os.listdir(path + '\\' + f2 + '\\' + f3)
            for f4 in files4:
                if '.mat' in f4:
                    print('dataProcessOfFixedFiles-' + f2 + '-' + f3 + '-' + f4)
                    # f4:1_1.mat 2_2.mat...
                    result = DataCalculate(path + '\\' + f2 + '\\' + f3 + '\\' + f4)

                    '''
                    经过去噪等算法处理之后的数据
                    '''
                    ap = result[0]
                    p = result[1]

                    N = result[2]
                    label = [l] * N

                    # 拼接后的数据
                    x = np.concatenate((ap.reshape(N, -1), p.reshape(N, -1)), axis=1)

                    scio.savemat(targetpath + f2 + '-' + f3 + '-' + f4,
                                 {'x': x, 'y': label})

    print('success:' + path)


def dataProcessOfOpenAndSemiFiles(path, targetpath):  # path:.../new targetpath:../demo.tfrecords

    files3 = os.listdir(path)

    for f3 in files3:
        # f3:1 2 3...
        files4 = os.listdir(path + '\\' + f3)
        l = int(f3) - 1
        for f4 in files4:
            if '.mat' in f4:
                print(
                    'dataProcessOfOpenAndSemiFiles-' + path + '-' + f3 + '-' + f4)
                # f4:1_1.mat 2_2.mat...
                result = DataCalculate(path + '\\' + f3 + '\\' + f4)
                ap = result[0]
                p = result[1]
                N = result[2]
                label = [l] * N

                scio.savemat(targetpath + f3 + '-' + f4,
                             {'x': np.concatenate((ap.reshape(N, -1), p.reshape(N, -1)), axis=1), 'y': label})

    print('success:' + path)


if __name__ == '__main__':
    path = '../new'
    tf_path = '../Data/csi_data.tfrecords'
    mat_path = '/Users/dmrfcoder/Documents/eating/1/eating_1_1.mat'

    fix_path = 'E:\\yczhao Data\\new\\fixed'
    fix_target_path = 'E:\\yczhao Data\\new-2-mat\\'
    # dataProcessOfFixedFiles(fix_path,fix_target_path)

    open_path = 'E:\\yczhao Data\\new\\open'
    open_target_path = 'E:\\yczhao Data\\new-2-mat1\\'
    # dataProcessOfOpenAndSemiFiles(open_path,open_target_path)

    semi_path = 'E:\\yczhao Data\\new\\semi'
    semi_target_path = 'E:\\yczhao Data\\new-2-mat2\\'
    # dataProcessOfOpenAndSemiFiles(semi_path,semi_target_path)

    threadsPool = [
        Thread(target=dataProcessOfFixedFiles, args=(fix_path, fix_target_path)),
        # Thread(target=dataProcessOfOpenAndSemiFiles, args=(open_path, open_target_path)),
        # Thread(target=dataProcessOfOpenAndSemiFiles,args=(semi_path,semi_target_path))
    ]
    for thread in threadsPool:
        thread.start()

    for thread in threadsPool:
        thread.join()
