# -*_coding:utf-8-*-
import os

import scipy.io as scio

def merge(path,target_path):
    files=os.listdir(path)
    for f1 in files:
        mat_data = scio.loadmat(path + '\\' + f1)
        x = mat_data['x']
        y = mat_data['y']


