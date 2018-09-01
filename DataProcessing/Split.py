import os

import scipy.io as scio

from Util.WriteHd5Util import writeToH5
from Util.loadHd import bean

import random


def split(sourcePath, targetPath, fragmentLength=200):
    files = os.listdir(sourcePath)
    x = []
    y = []
    for f1 in files:
        data = scio.loadmat(sourcePath + '\\' + f1)
        tempx = data['x']
        tempy = data['y']

        x.append(tempx)
        y.append(tempy)

    data = []
    l = len(y)
    for i in range(l):
        b = bean(x[i], y[i])
        data.append(b)

    random.shuffle(data)

    writeToH5(data, targetPath)


if __name__ == '__main__':
    fixed_path = 'F:\\csi\\AfterNormalizeData\\fixed\\'
    fixed_hd_target = 'F:\\csi\\AfterSplit200\\fixed.h5'

    split(fixed_path, fixed_hd_target, 200)
