# -*-coding:utf-8-*-

import tensorflow as tf
import numpy as np

from DlTrain.CNN import CNN
from DlTrain.Data import Data
from DlTrain.LSTM import LSTM
from DlTrain.Parameters import lstmTimeStep, lstmInputDimension, baseIr, valIterations, trainHd5Path, \
    trainBatchSize, trainReallyTxtPath, trainingIterations, trainPredictionTxtPath, valHd5Path, valBatchSize, \
    valPredictionTxtPath, valReallyTxtPath, pbPath, accuracyFilePath, maxAccuracyFilePath, valPerTrainIterations, \
    trainLogPath, valLogPath

lstmInput = tf.placeholder(tf.float32, shape=[None, lstmTimeStep * lstmInputDimension], name='inputLstm')
Label = tf.placeholder(tf.int32, shape=[None, ], name='Label')

cnnInput = LSTM(lstmInput)
cnnOutput = CNN(cnnInput)

with tf.name_scope('baseIr'):
    tf.summary.scalar('baseIr', baseIr)  # 写入tensorboard中的EVENTS

with tf.name_scope('Loss'):
    loss = tf.losses.sparse_softmax_cross_entropy(labels=Label, logits=cnnOutput)
    tf.summary.scalar('Loss', loss)

trainOp = tf.train.GradientDescentOptimizer(learning_rate=baseIr).minimize(loss)

predictionLabels = tf.cast(tf.argmax(cnnOutput, 1), tf.int32, name='PredictionLabels')

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

    trainLogWriter = tf.summary.FileWriter(trainLogPath, sess.graph)
    valLogWriter = tf.summary.FileWriter(valLogPath, sess.graph)

    trainData = Data(trainHd5Path, isTestCode)
    valDdata = Data(valHd5Path, isTestCode)

    for step in range(trainingIterations + 1):

        X, Y = trainData.getNextAutoShuffleBatch(trainBatchSize)

        sess.run(trainOp, feed_dict={lstmInput: X, Label: Y})
        if step % valPerTrainIterations == 0:
            valX, valY = valDdata.getNextAutoShuffleBatch(valBatchSize)
            valLoss, valAccuracy = sess.run([loss, Accuracy], feed_dict={lstmInput: valX, Label: valY})
            print('step:%d, trainLoss:%f, trainAccuracy:%f' % (step, valLoss, valAccuracy))
            valSummary, _ = sess.run([merged, trainOp], feed_dict={lstmInput: X, Label: Y})
            valLogWriter.add_summary(valSummary, step)

        trainSummary, _ = sess.run([merged, trainOp], feed_dict={lstmInput: X, Label: Y})
        trainLogWriter.add_summary(trainSummary, step)

    constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["PredictionLabels"])

    with tf.gfile.FastGFile(pbPath, mode='wb') as f:
        f.write(constant_graph.SerializeToString())

    if not isTestCode:
        trainLogWriter.close()
        valLogWriter.close()


else:
    output_graph_def = tf.GraphDef()
    with open(pbPath, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

    valPbLstmInput = sess.graph.get_tensor_by_name("inputLstm:0")
    print(valPbLstmInput)
    valPbLabel = sess.graph.get_tensor_by_name("Label:0")
    print(valPbLabel)
    valPbPredictionLabels = sess.graph.get_tensor_by_name("PredictionLabels:0")
    print(valPbPredictionLabels)

    correctPbPrediction = tf.equal(valPbPredictionLabels, valPbLabel)

    valPbAccuracy = tf.reduce_mean(tf.cast(correctPbPrediction, tf.float32))

    if isWriteFlag:
        valPredictionTxtFile = open(valPredictionTxtPath, 'wb')
        valReallyTxtFile = open(valReallyTxtPath, 'wb')

    data = Data(valHd5Path, isTestCode)

    for step in range(valIterations + 1):
        X, Y = data.getNextAutoShuffleBatch(valBatchSize)

        valAccuracy = sess.run(valPbAccuracy, feed_dict={valPbLstmInput: X, valPbLabel: Y})
        if isWriteFlag:
            np.savetxt(valReallyTxtFile, Y)
            np.savetxt(valPredictionTxtFile,
                       sess.run(valPbPredictionLabels, feed_dict={valPbLstmInput: X, valPbLabel: Y}))

        print('valAccuracy:%f' % (valAccuracy))

    if isWriteFlag:
        valPredictionTxtFile.close()
        valReallyTxtFile.close()

sess.close()
