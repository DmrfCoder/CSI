# -*-coding:utf-8-*-

'''
对Amplitude使用加权移动平均法进行去噪处理(说白了就是平滑处理吗妈的)
设置m为100

用法:Amplitude=WeightMoveAverage(Amplitude, N, m=100)

'''


def WeightMoveAverage(Amplitude, N, m=100):  # N为Amplitude的长度
    Sum = MitemSum(m)
    SumReciprocal = 1 / Sum

    for t in range(m - 1, N):
        SumAmplitude = 0
        for j in range(t - m + 1, t + 1):  # t-m+1~t,因为range右边为开,所以+1
            SumAmplitude = SumAmplitude + Amplitude[j] * (j % m + 1)

        Amplitude[t] = SumReciprocal * SumAmplitude

    return Amplitude


def MitemSum(m):
    sum = 0
    for i in range(1, m + 1):
        sum = sum + i
    return sum
