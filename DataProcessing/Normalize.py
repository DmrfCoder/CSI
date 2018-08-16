import os
import scipy.io as scio
import numpy as np

def Normalize(data):

    mx = max(data)
    mn = min(data)

    if mx<=1:
        if mn>=-1:
            return data

    m = np.mean(data)
    return [(float(i) - m) / (mx - mn) for i in data]


def NormalizeFiles(path,target_path):
    files1=os.listdir(path)
    for f1 in files1:
        data = scio.loadmat(path+'\\'+f1)
        x = data['x']
        y = data['y']
        l=len(x)

        for i in range(0,l):
            x[i]=Normalize(x[i])
            print(i)


        print('success:'+path)
        scio.savemat(target_path + f1,
                     {'x':x, 'y': y})




if __name__=='__main__':
    path='E:\\yczhao Data\\new-2-mat2'
    target_path='F:\\csi\\new-2-mat2\\'

    NormalizeFiles(path,target_path)




