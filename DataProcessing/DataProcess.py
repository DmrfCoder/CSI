# -*-coding:utf-8-*-
import os
import time

from threading import Thread
import scipy.io as scio

import numpy as np

from DataProcessing.CsiToAmplitudeAndPhase import getAmplitudesAndPhasesLength
from DataProcessing.DataCalculate import DataCalculate
from DataProcessing.Normalize import Normalize


def dataProcessOfFixedFiles(path, path_p, path_ap, path_p2, path_ap2, doneN, totalN,
                            begin_time):  # path:.../new targetpath:../demo.tfrecords

    files2 = os.listdir(path)



    for f2 in files2:
        # f2:eatting settig ...

        files3 = os.listdir(path + '/' + f2)
        for f3 in files3:
            # f3:1 2 3...
            l = int(f3) - 1
            files4 = os.listdir(path + '/' + f2 + '/' + f3)

            for f4 in files4:
                if '.mat' in f4:
                    # f4:1_1.mat 2_2.mat...

                    result = DataCalculate(path + '/' + f2 + '/' + f3 + '/' + f4)


                    '''
                    经过去噪等算法处理之后的数据
                    '''
                    ap = result[0]
                    p = result[1]
                    ap2 = result[2]
                    p2 = result[3]



                    N = result[4]
                    label = [l] * N

                    ap = ap.reshape(N, -1)
                    ap2 = ap2.reshape(N, -1)
                    p = p.reshape(N, -1)
                    p2 = p2.reshape(N, -1)

                    # 拼接后的数据
                    # x = np.concatenate((ap.reshape(N, -1), p.reshape(N, -1)), axis=1)

                    # lenght = x.shape[0]

                    for i in range(N):
                        ap[i] = Normalize(ap[i])
                        p[i] = Normalize(p[i])
                        ap2[i] = Normalize(ap2[i])
                        p2[i] = Normalize(p2[i])

                    scio.savemat(path_ap + '/' + f2 + '-' + f3 + '-' + f4, {'x': ap, 'y': label})
                    scio.savemat(path_ap2 + '/' + f2 + '-' + f3 + '-' + f4, {'x': ap2, 'y': label})
                    scio.savemat(path_p + '/' + f2 + '-' + f3 + '-' + f4, {'x': p, 'y': label})
                    scio.savemat(path_p2 + '/' + f2 + '-' + f3 + '-' + f4, {'x': p2, 'y': label})

                    doneN += N

                    now_time = time.time()
                    secod = round(now_time - begin_time, 2)
                    fenzhong = int(secod / 60)
                    secod = round(secod - fenzhong * 60, 2)
                    xiaoshi = int(fenzhong / 60)
                    fenzhong = fenzhong - xiaoshi * 60
                    persent = round((doneN / totalN) * 100, 2)

                    str1 = '已进行：' + str(persent) + '% 用时：' + str(xiaoshi) + '小时' + str(fenzhong) + '分钟' + str(
                        secod) + '秒 预计还需：'

                    secod = int((secod * totalN) / doneN)
                    fenzhong = int(secod / 60)
                    secod = secod - fenzhong * 60
                    xiaoshi = int(fenzhong / 60)
                    fenzhong = fenzhong - xiaoshi * 60

                    str2 = str(xiaoshi) + '小时' + str(fenzhong) + '分钟' + str(secod) + '秒'
                    print str1 + str2 + 'dataProcessOfFixedFiles-' + f2 + '-' + f3 + '-' + f4

        print 'success:' + path


def dataProcessOfOpenAndSemiFiles(path, path_p, path_ap, path_p2,
                                  path_ap2, doneN, totalN,
                                  begin_time):  # path:.../new targetpath:../demo.tfrecords

    files3 = os.listdir(path)

    for f3 in files3:
        # f3:1 2 3...

        files4 = os.listdir(path + '/' + f3)

        l = int(f3) - 1
        for f4 in files4:
            if '.mat' in f4:
                # f4:1_1.mat 2_2.mat...
                result = DataCalculate(path + '/' + f3 + '/' + f4)

                ap = result[0]
                p = result[1]
                ap2 = result[2]
                p2 = result[3]



                N = result[4]
                label = [l] * N

                ap = ap.reshape(N, -1)
                ap2 = ap2.reshape(N, -1)
                p = p.reshape(N, -1)
                p2 = p2.reshape(N, -1)

                # 拼接后的数据
                # x = np.concatenate((ap.reshape(N, -1), p.reshape(N, -1)), axis=1)

                # lenght = x.shape[0]

                for i in range(N):
                    ap[i] = Normalize(ap[i])
                    p[i] = Normalize(p[i])
                    ap2[i] = Normalize(ap2[i])
                    p2[i] = Normalize(p2[i])

                scio.savemat(path_ap + '/' + f3 + '-' + f4, {'x': ap, 'y': label})
                scio.savemat(path_ap2 + '/' + f3 + '-' + f4, {'x': ap2, 'y': label})
                scio.savemat(path_p + '/' + f3 + '-' + f4, {'x': p, 'y': label})
                scio.savemat(path_p2 + '/' + f3 + '-' + f4, {'x': p2, 'y': label})

                doneN += N

                now_time = time.time()
                secod = round(now_time - begin_time, 2)
                fenzhong = int(secod / 60)
                secod = round(secod - fenzhong * 60, 2)
                xiaoshi = int(fenzhong / 60)
                fenzhong = fenzhong - xiaoshi * 60
                persent = round((doneN / totalN) * 100, 2)

                str1 = '已进行：' + str(persent) + '% 用时：' + str(xiaoshi) + '小时' + str(fenzhong) + '分钟' + str(
                    secod) + '秒 预计还需：'

                secod = int((secod * totalN) / doneN)
                fenzhong = int(secod / 60)
                secod = secod - fenzhong * 60
                xiaoshi = int(fenzhong / 60)
                fenzhong = fenzhong - xiaoshi * 60

                str2 = str(xiaoshi) + '小时' + str(fenzhong) + '分钟' + str(secod) + '秒'
                print str1 + str2 + 'dataProcessOfOpenAndSemiFiles-' + path + '-' + f3 + '-' + f4

        print('success:' + path)


def getSemiOrOpenLength(path):
    N = 0
    files3 = os.listdir(path)

    for f3 in files3:
        # f3:1 2 3...

        files4 = os.listdir(path + '/' + f3)

        l = int(f3) - 1
        for f4 in files4:
            if '.mat' in f4:
                N += getAmplitudesAndPhasesLength(path + '/' + f3 + '/' + f4)

    return N


def getDataLength(fixed_path, open_path, semi_path):
    N = 0

    files2 = os.listdir(fixed_path)

    for f2 in files2:
        # f2:eatting settig ...

        files3 = os.listdir(fixed_path + '/' + f2)
        for f3 in files3:
            # f3:1 2 3...
            l = int(f3) - 1
            files4 = os.listdir(fixed_path + '/' + f2 + '/' + f3)

            for f4 in files4:
                if '.mat' in f4:
                    # print('dataProcessOfFixedFiles-' + f2 + '-' + f3 + '-' + f4)
                    # f4:1_1.mat 2_2.mat...
                    N += getAmplitudesAndPhasesLength(fixed_path + '/' + f2 + '/' + f3 + '/' + f4)
    print 'N of fixed:' + str(N)

    N2 = getSemiOrOpenLength(open_path)
    print 'N of open:' + str(N2)
    N3 = getSemiOrOpenLength(semi_path)
    print 'N of semi:' + str(N3)

    return N + N2 + N3


if __name__ == '__main__':
    '''
    N of fixed:39600968
    N of open:7476236
    N of semi:12373101
    total N:59450305
    '''
    path_fixed = '/media/xue/软件/CSI/RawMatData/fixed'
    path_open = '/media/xue/软件/CSI/RawMatData/open'
    path_semi = '/media/xue/软件/CSI/RawMatData/semi'

    # N = getDataLength(path_fixed, path_open, path_semi)
    # print 'total N:' + str(N)

    beagin_time = time.time()

    '''

    dataProcessOfFixedFiles(path='/media/xue/软件/CSI/RawMatData/fixed',
                            path_ap='/media/xue/软件/CSI/MatData/AmplitudeWithOutNoiseRemoval',
                            path_ap2='/media/xue/软件/CSI/MatData/AmplitudeWithNoiseRemoval',
                            path_p='/media/xue/软件/CSI/MatData/PhaseWithOutNoiseRemoval',
                            path_p2='/media/xue/软件/CSI/MatData/PhaseWithNoiseRemoval',
                            begin_time=beagin_time, doneN=0, totalN=59450305)
     '''

    '''
    dataProcessOfOpenAndSemiFiles(path='/media/xue/Data Storage/CSI/RawMatData/open',
                                  path_ap='/media/xue/Data Storage/CSI/MatData/AmplitudeWithOutNoiseRemoval',
                                  path_ap2='/media/xue/Data Storage/CSI/MatData/AmplitudeWithNoiseRemoval',
                                  path_p='/media/xue/Data Storage/CSI/MatData/PhaseWithOutNoiseRemoval',
                                  path_p2='/media/xue/Data Storage/CSI/MatData/PhaseWithNoiseRemoval',
                                  begin_time=beagin_time, doneN=39600968, totalN=59450305
                                  )
    '''

    dataProcessOfOpenAndSemiFiles(path='/media/xue/Data Storage/CSI/RawMatData/semi',
                                  path_ap='/media/xue/Data Storage/CSI/MatData/AmplitudeWithOutNoiseRemoval',
                                  path_ap2='/media/xue/Data Storage/CSI/MatData/AmplitudeWithNoiseRemoval',
                                  path_p='/media/xue/Data Storage/CSI/MatData/PhaseWithOutNoiseRemoval',
                                  path_p2='/media/xue/Data Storage/CSI/MatData/PhaseWithNoiseRemoval',
                                  begin_time=beagin_time, doneN=39600968 + 7476236, totalN=59450305
                                  )
