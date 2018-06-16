# Net Strcture
LSTM--->CNN-->FC-->Softmax
## LSTM
Input dimension:time*1*180*2
Units number:N=64
Output dimension time*N*1
## CNN
Two CNN blocks
each block contains filter and max pooling components

### first filter
#### cnn
6 filters
kernel size:5*5
stride:1*1
#### max pool:
size:2*2
stride:2*2

output:98*30*6

### second filter
#### cnn
10 filters
kernel size:5*3
stride:3*?
#### max pool
kernel size:
stride:
out put:32*10*10

## FC
Three layers
Input:32*10*10(flat to 3200*1)

+ 1000
+ 200
+ 5


 output:5*1
## Softmax
 5 units
