#coding:utf-8
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

trainingIterations = 10000  # 训练迭代次数
valIterations = 1000


baseIr = 0.1

valPerTrainIterations=4


'''
IO
'''
# Log path
trainLogPath = "../Log/open/train/"
valLogPath = "../Log/open/val/"
pbPath='../Model/open.pb'


accuracyFilePath='../Data/Open/Accuracy.txt'
maxAccuracyFilePath='../Data/Open/maxAccuracy.txt'
trainPredictionTxtPath = '../Data/Open/trainPredictionLabel.txt'
trainReallyTxtPath = '../Data/Open/trainReallyLabel.txt'

valPredictionTxtPath = '../Data/Open/valPredictionLabel.txt'
valReallyTxtPath = '../Data/Open/valReallyLabel.txt'

train_tf_path = '/data/after-split200/fixed/train.tfrecords'
val_tf_path = '/data/after-split200/fixed/val.tfrecords'

