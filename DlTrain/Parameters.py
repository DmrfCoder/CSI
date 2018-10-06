# coding:utf-8


fragmentLength = 1000

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
valIterations = 100

baseIr = 0.01

valPerTrainIterations = 4

'''
IO
'''



# Log path

logRoot = '/media/xue/Data Storage/CSI/Train/Log'

# pb path
pbRoot = '/media/xue/Data Storage/CSI/Train/Model'

# matrix path

matrixRoot = '/media/xue/Data Storage/CSI/Train/ConfusionMatrix'

tfRootPath = '/media/xue/Data Storage/CSI/TfRecordsData/'

