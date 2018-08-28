import random

from Util.loadHd import load
import numpy as np


def convertY(y_list):
    yListLength = len(y_list)
    yCoverted = np.zeros(shape=yListLength)
    for listItemIndex in range(0, yListLength):
        yCoverted[listItemIndex] = y_list[listItemIndex]

    return yCoverted


class Data:
    x = []
    y = []
    dataLength = 0
    isTest = False  # 是否为测试代码模式
    indexList = []

    def __init__(self, path, is_test):
        self.dataPath = path
        Data.isTest = is_test
        Data.loadData(self)

    def loadData(self):
        if Data.isTest:
            return
        data = load(self.dataPath)
        Data.x = data[0]
        Data.y = data[1]
        Data.dataLength = len(Data.y)

    def getNextManualShuffleBatch(self, batch_size):
        if self.isTest:
            X = np.random.random(size=(batch_size, 72000))
            Y = np.random.randint(0, 5, size=batch_size)
            return X, Y
        else:
            if len(self.indexList) < batch_size:
                self.indexList = list(range(0, Data.dataLength))

            randomIndexes = random.sample(range(0, len(self.indexList)), batch_size)
            X = []
            Y = []

            for randomIndex in randomIndexes:
                X.append(self.x[randomIndex])
                Y.append(self.y[randomIndex])

            sortedIndexes = sorted(randomIndexes, reverse=True)
            # 如果还抛异常用可以手动捕捉一下跳过
            for sortedIndex in sortedIndexes:
                self.indexList.pop(sortedIndex)

            X = np.reshape(X, newshape=(-1, 72000))
            Y = np.reshape(Y, newshape=(-1, 1))
            Y = convertY(Y)
            return X, Y

    def getNextAutoShuffleBatch(self, batch_size):
        if self.isTest:
            X = np.random.random(size=(batch_size, 72000))
            Y = np.random.randint(0, 5, size=batch_size)
            return X, Y
        else:

            if len(self.indexList) < batch_size:
                self.indexList = list(range(0, Data.dataLength))

            randomIndexes = random.sample(range(0, len(self.indexList)), batch_size)
            X = []
            Y = []

            for randomIndex in randomIndexes:
                X.append(self.x[randomIndex])
                Y.append(self.y[randomIndex])

            X = np.reshape(X, newshape=(-1, 72000))
            Y = np.reshape(Y, newshape=(-1, 1))
            Y = convertY(Y)
            return X, Y
