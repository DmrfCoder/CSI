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
trainingIterations = 30000  # 训练迭代次数
valIterations = 10000

baseIr = 0.0001

'''
IO
'''
# Log path
logPath = "../Log/"
ckptPath = '../ckpt_open/'

trainPredictionTxtPath = '../Data/trainPredictionLabel.txt'
trainReallyTxtPath = '../Data/trainReallyLabel.txt'

valPredictionTxtPath = '../Data/valPredictionLabel.txt'
valReallyTxtPath = '../Data/valReallyLabel.txt'

trainHd5Path = ''
valHd5Path = ''
