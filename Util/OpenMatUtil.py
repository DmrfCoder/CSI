#-*-coding:utf-8-*-
import scipy.io as scio
import numpy as np

data_path='/home/dmrfcoder/Document/CSI/DataSet/new/fixed/eating/1/eating_1_2.mat'
data = scio.loadmat(data_path)
# print type(data)
# print data
test_x=data['test_x']
print(test_x.shape)
print(test_x[0].shape)
a=test_x[0].reshape([360,1])
print(a.shape)
#a=np.where(test_x[0]==1)
#print(a[0][0])
