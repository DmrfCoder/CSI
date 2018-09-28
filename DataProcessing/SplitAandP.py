import os
import scipy.io as scio


def slpitAandP(path,pathA,pathP):
    files1 = os.listdir(path)

    for f1 in files1:#fixed open semi
        files2=os.listdir(path+'/'+f1)
        os.makedirs(pathA+'/'+f1)
        for f2 in files2:# .mat
            data = scio.loadmat(path+'/'+f1+'/'+f2)
            tempx = data['x']
            tempy = data['y']




