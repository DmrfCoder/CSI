# CSI Collection
## 得到amplitud和phase值
csi中的数据格式为a+bj即为复数,使用abs函数得到振幅(amplitude),使用phase函数得到相位(phase)

-  abs函数

> 求复数实部与虚部的平方和的算术平方根

-  phase函数

这样拿到了相位和振幅的原始数据(论文第5页B部分).


# Activity Recognition Model Construction

## Activity Recognition Preprocessing

### Butterworth
设置cut-off frequency 为 200Hz,这样可以滤掉200Hz以上的噪音,处理之后高频噪音消除.

### PCA
使用Butterworth可以滤掉高于200Hz的噪音,但是1~200Hz的噪音还在,使用PCA对这部分噪音进行去除:

- DC Component Removal

给csi减去一个恒定值以去除Component Removal(DC),这个恒定值由子载波(subcarrier)取平均得到?

- Principal Components

首先算出来相关矩阵Z = HT × H,然后通过对Z进行特征分解的处理得到特征向量Qi,然后根据hi = H × Q i算出来hi

- Smoothing

使用中值滤波进行平滑处理(5-point median filter)


DeepCount discards the first principal component h 1 and retains the next ten principal components to be used for feature extraction.


## Feature Extraction
使用多贝西D4小波变换(Daubechies D4 wavelet)[reference](https://blog.csdn.net/fengyu09/article/details/23207387) 来将PCA部分分解成从1Hz到200Hz的10个部分.

以128为时间窗平均系数,然后...

## Classification
提取到特征之后使用[隐马尔可夫模型(HMM)](https://zh.wikipedia.org/zh-hans/隐马尔可夫模型) 进行分类

## Monitor with Activity Recognition Model
...
# Deep Learning Model Construction

## Counting Model Preprocessing

- Amplitude Noise Removal

使用加权平均算法对振幅进行降噪,m设置为100

- Phase Sanitization
...

### Offline Training

