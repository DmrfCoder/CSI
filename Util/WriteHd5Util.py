import pandas as pd
import numpy as np

def write(data, path):
    f = pd.HDFStore(path, 'a')
    l = len(data)
    ndata=np.reshape(data,newshape=(-1,1))

    pddata=pd.DataFrame(ndata)
    f.append('data',pddata)

    f.close()
    print('success')
