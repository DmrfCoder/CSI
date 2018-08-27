# -*_coding:utf-8-*-
import os

import scipy.io as scio
import tensorflow as tf
import numpy as np


# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串类型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

list_x=[]
list_y=[]

list_x_val=[]
list_y_val=[]


def DoConvert(path_mat,new_path_mat,slice_len=200):




    files1=os.listdir(path_mat)

    for f1 in files1:
        mat_data = scio.loadmat(path_mat+'\\'+f1)
        x = mat_data['x']
        y =mat_data['y']
        label=y[0][0]
        item_count = y.shape[1]
        '''
        一共有item_count数量的数据,x的数据类型为float64,每一个x的维度为1*360
        已经归一化处理过
        '''
        item_count = int(item_count / slice_len)

        new_x=[]
        new_y=[]
        new_x_val=[]
        new_y_val=[]

        for i in range(0, item_count):
            data_raw = x[i * slice_len:(i + 1) * slice_len]
            if i/item_count>=0.8:
                print('doing_val:' + f1 + ' ' + str(i) + ' label:' + str(label))

                new_x_val.append(data_raw)
                new_y_val.append(label)

            else:
                print('doing:' + f1 + ' ' + str(i) + ' label:' + str(label))
                new_x.append(data_raw)
                new_y.append(label)

        scio.savemat(new_path_mat+'\\'+f1,{'x':new_x, 'y': new_y})
        scio.savemat(new_path_mat+'\\val_'+f1,{'x':new_x_val, 'y': new_y_val})


    print('success')



if __name__ == '__main__':
    path_mat='F:\\csi\\open'
    new_path_mat='F:\\csi\\open2'


    DoConvert(path_mat,new_path_mat )