import random

import numpy as np
import pandas as pd

<<<<<<< HEAD
from Util.WriteHd5Util import writeToH5
=======
from Util.WriteHd5Util import write
>>>>>>> c7c16b06acb9e61b60dd9bfbe34bf7628c81935b


def convertY(y_list):
    yListLength = len(y_list)
    yCoverted = np.zeros(shape=yListLength)
    for listItemIndex in range(0, yListLength):
        yCoverted[listItemIndex] = y_list[listItemIndex]

    return yCoverted


<<<<<<< HEAD

=======
class bean:

    def __init__(self, x, y):
        self.x = x
        self.y = y
>>>>>>> c7c16b06acb9e61b60dd9bfbe34bf7628c81935b


def load(path, path2):
    # f = pd.HDFStore(path, 'r')

    # x=f['x'].values
    # y=f['y'].values
    x = np.random.random(size=(2, 200, 360))
    x[0][0][0] = 0
    x[1][0][0] = 99
    x[0][199][359] = 88
    x[1][199][359] = 101
    y = [1, 0]
    # y=convertY(y)
    # final_x = np.reshape(x, (-1, 200, 360))
    final_x = x  # np.reshape(x, (-1, 72000))
    l = len(y)
    data = []
    for i in range(l):
        b = bean(final_x[i], y[i])
        data.append(b)

    random.shuffle(data)
<<<<<<< HEAD
    writeToH5(data, path2)
=======
    write(data, path2)
>>>>>>> c7c16b06acb9e61b60dd9bfbe34bf7628c81935b

    return (final_x, y)


def open():
    f = pd.HDFStore('open_val_sf.h5', 'r')
#    x = f['x'].values
    y = f['y'].values
    x = []

    l = len(y)

    for i in range(l):
        tx = f[str(i)].values
        x.append(tx)
    # x = np.reshape(x, (-1, 200,360))
    print(x[0][0][0],
          x[1][0][0],
          x[0][199][359],
          x[1][199][359])


# load('F:\csi\openh5\\open_val.h5','open_val_sf.h5')
<<<<<<< HEAD
=======
open()
>>>>>>> c7c16b06acb9e61b60dd9bfbe34bf7628c81935b
