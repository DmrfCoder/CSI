#-*-coding:utf-8-*-
import scipy.io as scio
import numpy as np


data_path='/home/dmrfcoder/Document/CSI/DataSet/new/fixed/eating/1/eating_1_2.mat'

# print type(data)
# print data


a=np.random.random(size=[2,3,6])
b=np.random.random(size=[2,3,6])
print(a)
print(b)

d=a.reshape(2,-1)
e=b.reshape(2,-1)
c=np.concatenate((a.reshape(2,-1), b.reshape(2,-1)))
print(c)



dataNew = open('demo.mat','a')

for i in range(10):
    scio.savemat("demo.mat", {'0': i,'1':i},appendmat=True)
    scio.savemat("demo.mat", {'2': i,'3':i},appendmat=True)


data = scio.loadmat("demo.mat")
test_x=data['A']
test_y=data['B']
print(test_x.shape)
print(test_x[0].shape)
a=test_x[0].reshape([360,1])
print(a.shape)
#a=np.where(test_x[0]==1)
#print(a[0][0])


