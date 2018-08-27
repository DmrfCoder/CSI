# -*-coding:utf-8-*-

import tensorflow as tf
import numpy as np

from DlTrain.CNN import CNN
from DlTrain.Data import Data
from DlTrain.LSTM import LSTM
from DlTrain.Parameters import lstmTimeStep, lstmInputDimension, baseIr, ckptPath, valIterations, trainHd5Path, \
    trainBatchSize, trainReallyTxtPath, logPath, trainingIterations, trainPredictionTxtPath, valHd5Path, valBatchSize, \
    valPredictionTxtPath, valReallyTxtPath

lstmInput = tf.placeholder(tf.float32, shape=[None, lstmTimeStep * lstmInputDimension], name='inputLstm')
Label = tf.placeholder(tf.int32, shape=[None, ])

cnnInput = LSTM(lstmInput)
cnnOutput = CNN(cnnInput)

with tf.name_scope('baseIr'):
    tf.summary.scalar('baseIr', baseIr)  # 写入tensorboard中的EVENTS

with tf.name_scope('Loss'):
    loss = tf.losses.sparse_softmax_cross_entropy(labels=Label, logits=cnnOutput)
    tf.summary.scalar('Loss', loss)

trainOp = tf.train.GradientDescentOptimizer(learning_rate=baseIr).minimize(loss)

predictionLabels = tf.cast(tf.argmax(cnnOutput, 1), tf.int32)

correctPrediction = tf.equal(predictionLabels, Label)

with tf.name_scope('Accuracy'):
    Accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
    tf.summary.scalar('Accuracy', Accuracy)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

isTestMode = False  # 是否是验证阶段
isTestCode = True  # 是否是测试代码模式（产生随机数据）
isWriteFlag = True  # 是否将label写入文件
saver = tf.train.Saver(max_to_keep=3)
merged = tf.summary.merge_all()

if not isTestMode:

    if isWriteFlag:
        trainPredictionTxtFile = open(trainPredictionTxtPath, 'wb')
        trainReallyTxtFile = open(trainReallyTxtPath, 'wb')

    accuracyFile = open(ckptPath + 'Accuracy.txt', 'w')
    maxAccuracyFile = open(ckptPath + 'maxAccuracy.txt', 'w')

    logWriter = tf.summary.FileWriter(logPath, sess.graph)
    maxAccuracy = 0

    data = Data(trainHd5Path, isTestCode)

    for step in range(trainingIterations + 1):

        X, Y = data.getNextAutoShuffleBatch(trainBatchSize)

        sess.run(trainOp, feed_dict={lstmInput: X, Label: Y})
        if isWriteFlag:
            np.savetxt(trainReallyTxtFile, Y)
            np.savetxt(trainPredictionTxtFile, sess.run(predictionLabels, feed_dict={lstmInput: X, Label: Y}))

        TrainLoss, TrainAccuracy = sess.run([loss, Accuracy], feed_dict={lstmInput: X, Label: Y})
        print('step:%d, trainLoss:%f, trainAccuracy:%f' % (step, TrainLoss, TrainAccuracy))

        accuracyFile.write(str(step + 1) + ', trainAccuracy: ' + str(TrainAccuracy) + '\n')
        if TrainAccuracy > maxAccuracy:
            maxAccuacy = TrainAccuracy
            maxAccuracyFile.write('maxAccuracy: ' + str(maxAccuacy) + '\n')

            saver.save(sess, ckptPath + 'csi.ckpt', global_step=step + 1)

        summary, _ = sess.run([merged, trainOp], feed_dict={lstmInput: X, Label: Y})
        logWriter.add_summary(summary, step)

    accuracyFile.close()
    maxAccuracyFile.close()

    if not isTestCode:
        logWriter.close()

    if isWriteFlag:
        trainPredictionTxtFile.close()
        trainReallyTxtFile.close()

else:
    modelFile = tf.train.latest_checkpoint(ckptPath)
    saver.restore(sess, modelFile)

    if isWriteFlag:
        valPredictionTxtFile = open(valPredictionTxtPath, 'wb')
        valReallyTxtFile = open(valReallyTxtPath, 'wb')

    data = Data(valHd5Path, isTestCode)

    for step in range(valIterations + 1):
        X, Y = data.getNextAutoShuffleBatch(valBatchSize)

        valLoss, valAccuracy = sess.run([loss, Accuracy], feed_dict={lstmInput: X, Label: Y})
        if isWriteFlag:
            np.savetxt(valReallyTxtFile, Y)
            np.savetxt(valPredictionTxtFile, sess.run(predictionLabels, feed_dict={lstmInput: X, Label: Y}))

        print('valLoss:%f, valAccuracy:%f' % (valLoss, valAccuracy))

    if isWriteFlag:
        valPredictionTxtFile.close()
        valReallyTxtFile.close()

sess.close()
