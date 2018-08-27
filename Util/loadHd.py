import numpy as np
import pandas as pd

def load(path):
    f = pd.HDFStore(path, 'r')
    x=f['x'].values
    y=f['y'].values
    final_x=np.reshape(x,(-1,200,360))
    return (final_x,y)

