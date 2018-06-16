# Net Strcture
LSTM--->CNN-->FC-->Softmax
## LSTM
Input dimension:time_1_180_2

Units number:N=64

Output dimension time_N_1
## CNN
Two CNN blocks
each block contains filter and max pooling components

### first filter
#### cnn
input dimision:200_64

so the time is 200 ?

6 filters

kernel size:5_5

stride:1_1
#### max pool:
size:2_2
stride:2_2

output:98_30_6

### second filter
#### cnn
10 filters

kernel size:5_3

stride:3*3
#### max pool
kernel size:1

stride:1

del this max pool?

out put:32_10_10

## FC
Three layers

Input:32_10_10(flat to 3200*1)

+ 1000
+ 200
+ 5


 output:5_1
## Softmax
 5 units
