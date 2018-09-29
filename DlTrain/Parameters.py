# coding:utf-8
from DlTrain.Train import train

fragmentLength = 1000

"""
LSTM
"""

lstmTimeStep = 1000
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

rootType = ['AmplitudeWithout_PhaseWith', 'AmplitudeWithOut_PhaseWithout', 'AmplitudeWith_PhaseWith',
            'AmplitudeWith_PhaseWithout', 'OnlyAmplitude', 'OnlyPhase']

# Log path

logRoot = '/media/xue/Data Storage/CSI/Train/Log'

# pb path
pbRoot = '/media/xue/Data Storage/CSI/Train/Model'

# matrix path

matrixRoot = '/media/xue/Data Storage/CSI/Train/ConfusionMatrix'

tfRootPath = '/media/xue/Data Storage/CSI/TfRecordsData/'

for i in range(6):
    if i < 5:
        train(rootType=rootType[i], which='fixed', baseIr=0.2)
        train(rootType=rootType[i], which='open', baseIr=0.15)
        train(rootType=rootType[i], which='semi', baseIr=0.1)
    else:
        train(rootType=rootType[i], which='fixed', baseIr=0.2, InputDimension=180)
        train(rootType=rootType[i], which='open', baseIr=0.15, InputDimension=180)
        train(rootType=rootType[i], which='semi', baseIr=0.1, InputDimension=180)
