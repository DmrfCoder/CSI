# -*-coding:utf-8-*-
import scipy.io as scio


def WriteListToMatFile(list, target_path):
    scio.savemat(target_path, {'key', list})



    
