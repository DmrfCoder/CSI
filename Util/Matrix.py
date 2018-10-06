# -*-coding:utf-8-*-
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

label = [0, 1, 2, 3, 4]
label2 = [4, 3, 2, 1, 0]


def drawMatrix(reallyTxtPath, predictionTxtPath, matrixPath):
    a = []
    b = []

    y_true = np.loadtxt(reallyTxtPath)
    y_pred = np.loadtxt(predictionTxtPath)

    # with open(reallyTxtPath, 'r') as f:
    #     for line in f:
    #         data = line.split()
    #         a.append(int(data[0][0]))
    #
    # with open(predictionTxtPath, 'r') as f:
    #     for line in f:
    #         data = line.split()
    #         b.append(int(data[0][0]))

    y_true=np.append(y_true, label)
    y_pred=np.append(y_pred, label)

    cm = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(9, 8), dpi=120)

    ind_array = np.array(label)
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.5:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='white', fontsize=35, va='center', ha='center')
        else:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='black', fontsize=35, va='center', ha='center')

    # offset the tick

    tick_marks = np.array(range(5)) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    labels = [1, 2, 3, 4, 5]
    plt.xticks(tick_marks - 0.5, labels, fontsize=35)
    plt.yticks(tick_marks - 0.5, labels, fontsize=35)

    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.binary)
    # show confusion matrix
    plt.savefig(matrixPath, format='png', dip=(420, 317))
    plt.close()
    #plt.show()


