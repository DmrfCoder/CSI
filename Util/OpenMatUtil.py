#-*-coding:utf-8-*-
import scipy.io as scio
import numpy as np

data_path='E:\\yczhao Data\\new-2-mat1\\1-open_1_2.mat'
#data = scio.loadmat(data_path)
# print type(data)
# print data

data = scio.loadmat(data_path)
x=data['x']
y=data['y']

N=2
ap=np.array([[1,2,3],[4,5,6]])
p=np.array([[7,8,9],[10,11,12]])
label=N*[1]

d= np.concatenate((ap.reshape(N, -1), p.reshape(N, -1)),axis=0)
d1= np.concatenate((ap.reshape(N, -1), p.reshape(N, -1)),axis=1)
scio.savemat("demo.mat",
             {'x': np.concatenate((ap.reshape(N, -1), p.reshape(N, -1)), axis=0), 'y': label})


data = scio.loadmat('demo.mat')
test_x=data['x']
test_y=data['y']
print(test_x.shape)
print(test_x[0].shape)
a=test_x[0].reshape([360,1])
print(a.shape)
#a=np.where(test_x[0]==1)
#print(a[0][0])
