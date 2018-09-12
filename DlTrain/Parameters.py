# coding:utf-8

"""
LSTM
"""

lstmTimeStep = 200
lstmHiddenUnits = 64
lstmLayerNum = 1
lstmInputDimension = 360

'''
the parameters of global
'''
classes = 5

trainBatchSize = 64

valBatchSize = trainBatchSize

trainingIterations = 3000  # 训练迭代次数
valIterations = 100

baseIr = 0.01

valPerTrainIterations = 4

'''
IO
'''
# Log path
which = "Open"

trainLogPath = '../Log/' + which + '/train/'
valLogPath = '../Log/' + which + '/val/'

pbPath = '../Model/' + which + '.pb'

accuracyFilePath = '../Data/' + which + '/Accuracy.txt'
maxAccuracyFilePath = '../Data/' + which + '/maxAccuracy.txt'
trainPredictionTxtPath = '../Data/' + which + '/trainPredictionLabel.txt'
trainReallyTxtPath = '../Data/' + which + '/trainReallyLabel.txt'

valPredictionTxtPath = '../Data/' + which + '/valPredictionLabel.txt'
valReallyTxtPath = '../Data/' + which + '/valReallyLabel.txt'

train_tf_path = '/data/after-split200/' + which + '/train.tfrecords'
val_tf_path = '/data/after-split200/' + which + '/val.tfrecords'
