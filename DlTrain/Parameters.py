# coding:utf-8
import os


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
#logRoot = '/home/xue/Log'

# pb path
pbRoot = '/media/xue/Data Storage/CSI/Train/Model'

# matrix path

matrixRoot = '/media/xue/Data Storage/CSI/Train/ConfusionMatrix'

tfRootPath = '/media/xue/Data Storage/CSI/TfRecordsData/'


def sort(list):
    max = -1

    for l in list:
        if max < int(l):
            max = int(l)

    return max



def init_folder(rootType, which):
    folders_dict = {}

    # Log file init
    if not os.path.exists(logRoot + '/' + rootType):
        os.mkdir(logRoot + '/' + rootType)

    if not os.path.exists(logRoot + '/' + rootType + '/' + which):
        os.mkdir(logRoot + '/' + rootType + '/' + which)

    logfiles = os.listdir(logRoot + '/' + rootType + '/' + which)
    logsort = sort(logfiles)

    if logsort == -1:
        os.mkdir(logRoot + '/' + rootType + '/' + which + '/0')

        os.mkdir(logRoot + '/' + rootType + '/' + which + '/0/train')
        os.mkdir(logRoot + '/' + rootType + '/' + which + '/0/val')

        trainLogPath = logRoot + '/' + rootType + '/' + which + '/0/train/'
        valLogPath = logRoot + '/' + rootType + '/' + which + '/0/val/'

    else:

        intLastIndex = logsort
        intLastIndex += 1

        lastIndex = str(intLastIndex)
        os.mkdir(logRoot + '/' + rootType + '/' + which + '/' + lastIndex)
        os.mkdir(logRoot + '/' + rootType + '/' + which + '/' + lastIndex + '/train')
        os.mkdir(logRoot + '/' + rootType + '/' + which + '/' + lastIndex + '/val')
        trainLogPath = logRoot + '/' + rootType + '/' + which + '/' + lastIndex + '/train/'
        valLogPath = logRoot + '/' + rootType + '/' + which + '/' + lastIndex + '/val/'

    folders_dict['trainLogPath'] = trainLogPath
    folders_dict['valLogPath'] = valLogPath

    # Pb file init
    if not os.path.exists(pbRoot + '/' + rootType):
        os.mkdir(pbRoot + '/' + rootType)

    if not os.path.exists(pbRoot + '/' + rootType + '/' + which):
        os.mkdir(pbRoot + '/' + rootType + '/' + which)

    pbfiles = os.listdir(pbRoot + '/' + rootType + '/' + which)
    pbsort = sort(pbfiles)
    if pbsort == -1:
        os.mkdir(pbRoot + '/' + rootType + '/' + which + '/0')
        pbPath = pbRoot + '/' + rootType + '/' + which + '/0/model.pb'


    else:
        intLastIndex = pbsort
        intLastIndex += 1
        lastIndex = str(intLastIndex)
        os.mkdir(pbRoot + '/' + rootType + '/' + which + '/' + lastIndex)
        pbPath = pbRoot + '/' + rootType + '/' + which + '/' + lastIndex + '/model.pb'

    folders_dict['pbPath'] = pbPath

    # matrix file init
    if not os.path.exists(matrixRoot + '/' + rootType):
        os.mkdir(matrixRoot + '/' + rootType)

    if not os.path.exists(matrixRoot + '/' + rootType + '/' + which):
        os.mkdir(matrixRoot + '/' + rootType + '/' + which)

    matrixfiles = os.listdir(matrixRoot + '/' + rootType + '/' + which)
    masort = sort(matrixfiles)
    trainPredictionTxtPath = '/trainPredictionLabel.txt'
    trainReallyTxtPath = '/trainReallyLabel.txt'
    matrixPicturePath = '/confusionMatrix.png'

    if masort == -1:
        os.mkdir(matrixRoot + '/' + rootType + '/' + which + '/0')
        os.mkdir(matrixRoot + '/' + rootType + '/' + which + '/0/Picture')
        os.mkdir(matrixRoot + '/' + rootType + '/' + which + '/0/Txt')

        trainPredictionTxtPath = matrixRoot + '/' + rootType + '/' + which + '/0/Txt' + trainPredictionTxtPath
        trainReallyTxtPath = matrixRoot + '/' + rootType + '/' + which + '/0/Txt' + trainReallyTxtPath
        matrixPicturePath = matrixRoot + '/' + rootType + '/' + which + '/0/Picture' + matrixPicturePath

    else:
        intLastIndex = masort
        intLastIndex += 1
        lastIndex = str(intLastIndex)
        os.mkdir(matrixRoot + '/' + rootType + '/' + which + '/' + lastIndex)
        os.mkdir(matrixRoot + '/' + rootType + '/' + which + '/' + lastIndex + '/Picture')
        os.mkdir(matrixRoot + '/' + rootType + '/' + which + '/' + lastIndex + '/Txt')

        trainPredictionTxtPath = matrixRoot + '/' + rootType + '/' + which + '/' + lastIndex + '/Txt' + trainPredictionTxtPath
        trainReallyTxtPath = matrixRoot + '/' + rootType + '/' + which + '/' + lastIndex + '/Txt' + trainReallyTxtPath
        matrixPicturePath = matrixRoot + '/' + rootType + '/' + which + '/' + lastIndex + '/Picture' + matrixPicturePath

    folders_dict['trainPredictionTxtPath'] = trainPredictionTxtPath
    folders_dict['trainReallyTxtPath'] = trainReallyTxtPath
    folders_dict['matrixPicturePath'] = matrixPicturePath

    return folders_dict
