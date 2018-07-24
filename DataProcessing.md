# 输出处理过程
## 得到amplitud和phase值
csi中的数据格式为a+bj即为复数,使用abs函数得到振幅(amplitude),使用phase函数得到相位(phase)

-  abs函数

> 求复数实部与虚部的平方和的算术平方根

-  phase函数

这样拿到了相位和振幅的原始数据(论文第5页B部分).

## 对原始数据进行处理
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




