import os
import random
import pandas as pd
import scipy.io as scio
import numpy as np
import h5py
import pandas as pd


def read(path1,target):
    files=os.listdir(path1)
    d1=[]
    d2=[]
    #f = h5py.File('F:\\csi\\opennpy_x.txt', 'a')
    f = pd.HDFStore(target, 'a')

    isd=0
    for f1 in files:
        print(f1)
        data=scio.loadmat(path1+'\\'+f1)
        x=data['x']
        y=data['y'][0]



        l=len(x)
        for i in  range(0,l):
            temp_x=x[i]
            a = pd.DataFrame(temp_x)
            f.append('x',a)

        b=pd.DataFrame(y)
        f.append('y',b)



    f.close()


    return (d1,d2)

f=read('F:\\csi\\open2_val','F:\\csi\\open_val.h5')