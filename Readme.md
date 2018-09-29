# CSI

## Deep DeepCount

LSTM--->CNN-->FC-->Softmax
### LSTM

Input dimension:time_360

one layer

Units number:N=64

Output dimension time_N_1
### CNN

Two CNN blocks
each block contains filter and max pooling components

#### first filter

- cnn

input dimision:200_64

so the time is 200 ?

6 filters

kernel size:5_5

stride:1_1
- max pool:

size:2_2
stride:2_2

output:98_30_6

#### second filter

- cnn

10 filters

kernel size:5_3

stride:3*3
- max pool

kernel size:1

stride:1

del this max pool?

out put:32_10_10

### FC

Three layers

Input:32_10_10(flat to 3200*1)

+ 1000
+ 200
+ 5


 output:5_1
### Softmax

 5 units

## Data Processing

### Algorithm

- Amplitude Noise Removal

使用加权平均算法对振幅进行降噪,m设置为100

- Phase Sanitization

首先对原始phase数据(180--->6*30) unwrap,然后计算出every subcarrier的均值y,利用y和x:[0:Sub-1]进行线性拟合(linear fit),最终算出calibrated phase value 并返回.

### Code

- DataProcess

> 使用weight moving等算法对原始数据进行处理，得到净数据

- Normalize(已经在上一步进行了归一化处理)

> 对净数据进行归一化处理

经过以上两步处理得到fixed、open、semi三个文件夹下的数据文件夹，每个数据文件夹下的数据都是N×360的且已经做过归一化处理



## Comparative Experiment

在两个方面做两组对比实验：

- Only Amplitude(Without P)
- Only Phase(Without A)
- Without Amplitude noise removal  but with Phase noise removal(Without A)
- With Amplitude noise removal  but without Phase noise removal(Without P)
- Amplitude with noise removal and Phase with noise removal(With P*A)
- Amplitude without noise removal and Phase without noise removal(Raw Data)


基于此，数据集应有如下几种：

- Amplitude with noise removal 

- Phase with noise removal

- Amplitude without noise removal （原始数据就是，但是需要将小数据集拼接成一个数据集）

- Phase without noise removal （原始数据就是，但是需要将小数据集拼接成一个数据集）


使用以上四个数据集，组合成以下数据集进行训练：

- Amplitude without noise removal&Phase with noise removal

- Amplitude with noise removal&Phase without noise removal

- Amplitude with noise removal&Phase with noise removal

- Amplitude without noise removal&Phase without noise removal


另外使用另一个网络对如下数据集进行训练：

- Amplitude with noise removal

- Phase with noise removal


新网络的改进策略是讲原来的360统一换成180，切片长度和Units number不变，这样LSTM的输出维度就不变，这样CNN部分就不用修改。