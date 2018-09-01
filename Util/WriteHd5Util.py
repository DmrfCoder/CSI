import pickle

import pandas as pd
import numpy as np


def writeToH5(data, path):
    f = pd.HDFStore(path, 'a')
    l = len(data)

    y = []

    for i in range(l):
        tempx = data[i].x
        pdatax = pd.DataFrame(tempx)
        f.append(str(i), pdatax)

        tempy = data[i].y
        y.append(tempy)

    pdatay = pd.DataFrame(y)
    f.append('y', pdatay)

    f.close()
    print('success')
