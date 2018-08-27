#-*-coding:utf-8-*-
import scipy.io as scio
import numpy as np

<<<<<<< HEAD
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
=======
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
>>>>>>> ca38e2faf428cfaa66bb64e10c3d32227dae4e03
print(test_x.shape)
print(test_x[0].shape)
a=test_x[0].reshape([360,1])
print(a.shape)
#a=np.where(test_x[0]==1)
#print(a[0][0])


