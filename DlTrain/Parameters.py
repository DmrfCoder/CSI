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

trainBatchSize = 16
valBatchSize = trainBatchSize
trainingIterations = 10  # 训练迭代次数
valIterations = 10
valPerTrainIterations=10
baseIr = 0.0001

'''
IO
'''
# Log path
trainLogPath = "../Log/train/"
valLogPath = "../Log/val/"
pbPath='../Model/open.pb'


accuracyFilePath='../Data/Open/Accuracy.txt'
maxAccuracyFilePath='../Data/Open/maxAccuracy.txt'
trainPredictionTxtPath = '../Data/Open/trainPredictionLabel.txt'
trainReallyTxtPath = '../Data/Open/trainReallyLabel.txt'

valPredictionTxtPath = '../Data/Open/valPredictionLabel.txt'
valReallyTxtPath = '../Data/Open/valReallyLabel.txt'

trainHd5Path = ''
valHd5Path = ''
