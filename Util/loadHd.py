import random

import numpy as np
import pandas as pd

from Util.WriteHd5Util import write


def convertY(y_list):
    yListLength = len(y_list)
    yCoverted = np.zeros(shape=yListLength)
    for listItemIndex in range(0, yListLength):
        yCoverted[listItemIndex] = y_list[listItemIndex]

    return yCoverted

class bean:

    def __init__(self,x,y):
        self.x=x
        self.y=y


def load(path,path2):
    f = pd.HDFStore(path, 'r')

    x=f['x'].values
    y=f['y'].values
    y=convertY(y)
    #final_x = np.reshape(x, (-1, 200, 360))
    final_x = np.reshape(x, (-1,72000))
    l=len(y)
    data=[]
    for i in range(l):
        b=bean(final_x[i],y[i])
        data.append(b)

    random.shuffle(data)
    write(data,path2)


    return (final_x,y)

load('F:\csi\openh5\\open_val.h5','F:\csi\openh5\\open_val_sf.h5')
