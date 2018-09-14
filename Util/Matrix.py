# -*-coding:utf-8-*-
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# load labels.
#labels = ['A', 'B', 'C', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
#labels = ['A', 'B', 'C', 'F', 'G', 'H', 'I','J']

label= [0,1, 2, 3, 4]

#y_true = np.loadtxt('../Data/pc_re_label.txt')
#y_pred = np.loadtxt('../Data/pc_pr_label_tf.txt')

which='Open'
reallyTxtPath='../Data/' + which + '/valReallyLabel.txt'
predictionTxtPath='../Data/' + which + '/valPredictionLabel.txt'

#a = np.loadtxt(reallyTxtPath)
#b = np.loadtxt(predictionTxtPath)

a = []
b = []


with open(reallyTxtPath, 'r') as f:
    for line in f:
        data = line.split()
        a.append(int(data[0][0]))


with open(predictionTxtPath, 'r') as f:
    for line in f:
        data = line.split()
        b.append(int(data[0][0]))


y_true=a
y_pred=b




def plot_confusion_matrix(cm, cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    #plt.colorbar()
   # xlocations = np.array(range(len(labels)))
    #plt.xticks(xlocations, labels, rotation=90)
    #plt.yticks(xlocations, labels)
    #plt.ylabel('True label')
    #plt.xlabel('Predicted label')


cm = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(9, 8), dpi=120)
# set the fontsize of label.
# for label in plt.gca().xaxis.get_ticklabels():
#    label.set_fontsize(8)
# text portion
ind_array = np.array(label)
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    if c > 0.01:
        if c>0.5:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='white', fontsize=14, va='center', ha='center')
        else:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='black', fontsize=14, va='center', ha='center')



# offset the tick


tick_marks = np.array(range(5)) + 0.5

plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

labels=[1,2,3,4,5]
plt.xticks(tick_marks-0.5, labels)
plt.yticks(tick_marks-0.5, labels)



plot_confusion_matrix(cm_normalized)
# show confusion matrix
plt.savefig('../Data/'+which+'_confusion_matrix_val.png', format='png')
plt.show()