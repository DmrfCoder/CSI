# -*-coding:utf-8-*-

import tensorflow as tf
import numpy as np

from DlTrain.CNN import CNN
from DlTrain.Data import Data
from DlTrain.LSTM import LSTM

from DlTrain.Parameters import lstmTimeStep, lstmInputDimension, baseIr, valIterations, \
    trainBatchSize, trainReallyTxtPath, trainingIterations, trainPredictionTxtPath, valBatchSize, \
    valPredictionTxtPath, valReallyTxtPath, pbPath, accuracyFilePath, maxAccuracyFilePath, valPerTrainIterations, \
    trainLogPath, valLogPath, val_tf_path, train_tf_path
from Util.ReadAndDecodeUtil import read_and_decode

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

x_train, y_train = read_and_decode(train_tf_path)
num_threads = 3
min_after_dequeue_train = 10000

train_capacity_train = min_after_dequeue_train + num_threads * trainBatchSize

# 使用shuffle_batch可以随机打乱输入
train_x_batch, train_y_batch = tf.train.shuffle_batch([x_train, y_train],
                                                      batch_size=trainBatchSize, capacity=train_capacity_train,
                                                      min_after_dequeue=min_after_dequeue_train)

x_val, y_val = read_and_decode(val_tf_path)
num_threads = 3
min_after_dequeue_val = 2500

train_capacity_val = min_after_dequeue_val + num_threads * trainBatchSize

# 使用shuffle_batch可以随机打乱输入
val_x_batch, val_y_batch = tf.train.shuffle_batch([x_val, y_val],
                                                  batch_size=valBatchSize, capacity=train_capacity_val,
                                                  min_after_dequeue=min_after_dequeue_val)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
threads = tf.train.start_queue_runners(sess=sess)

isTestMode = False  # 是否是验证阶段
isTestCode = False  # 是否是测试代码模式（产生随机数据）

isWriteFlag = True  # 是否将label写入文件
saver = tf.train.Saver(max_to_keep=1)
merged = tf.summary.merge_all()

if not isTestMode:

    trainLogWriter = tf.summary.FileWriter(trainLogPath, sess.graph)
    valLogWriter = tf.summary.FileWriter(valLogPath, sess.graph)

    if isWriteFlag:
        valPredictionTxtFile = open(valPredictionTxtPath, 'wb')
        valReallyTxtFile = open(valReallyTxtPath, 'wb')

    for step in range(trainingIterations + 1):

        # X, Y = trainData.getNextManualShuffleBatch(trainBatchSize)
        X, Y = sess.run([train_x_batch, train_y_batch])
        X = np.reshape(X, newshape=(-1, 72000))

        sess.run(trainOp, feed_dict={lstmInput: X, Label: Y})
        if step % valPerTrainIterations == 0:
            valX, valY = sess.run([val_x_batch, val_y_batch])
            valX = np.reshape(valX, newshape=(-1, 72000))

            out_labels = sess.run(predictionLabels, feed_dict={lstmInput: valX, Label: valY})

            for i in range(valBatchSize):
                print 'really:%d prediction:%d' % (valY[i], out_labels[i])

            if isWriteFlag:
                np.savetxt(valReallyTxtFile, valY)
                np.savetxt(valPredictionTxtFile, out_labels)

            valLoss, valAccuracy = sess.run([loss, Accuracy], feed_dict={lstmInput: valX, Label: valY})
            print('step:%d, valLoss:%f, valAccuracy:%f' % (step, valLoss, valAccuracy))
            valSummary, _ = sess.run([merged, trainOp], feed_dict={lstmInput: X, Label: Y})
            valLogWriter.add_summary(valSummary, step)

        trainSummary, _ = sess.run([merged, trainOp], feed_dict={lstmInput: X, Label: Y})
        trainLogWriter.add_summary(trainSummary, step)

    constant_graph = tf.grap
    h_util.convert_variables_to_constants(sess, sess.graph_def, ["PredictionLabels"])

    if not isTestCode:
        trainLogWriter.close()
        valLogWriter.close()

    with tf.gfile.FastGFile(pbPath, mode='wb') as f:
        f.write(constant_graph.SerializeToString())




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

    if isWriteFlag:
        valPredictionTxtFile = open(valPredictionTxtPath, 'wb')
        valReallyTxtFile = open(valReallyTxtPath, 'wb')

    for step in range(valIterations + 1):

        valX, valY = sess.run([val_x_batch, val_y_batch])
        valX = np.reshape(valX, newshape=(-1, 72000))

        pb_out_labels = sess.run(valPbPredictionLabels, feed_dict={valPbLstmInput: valX, valPbLabel: valY})

        print('step:%d' % step)
        for i in range(valBatchSize):
            print 'really:%d prediction:%d' % (valY[i], pb_out_labels[i])

        if isWriteFlag:
            np.savetxt(valReallyTxtFile, valY)
            np.savetxt(valPredictionTxtFile, pb_out_labels)

    if isWriteFlag:
        valPredictionTxtFile.close()
        valReallyTxtFile.close()

    sess.close()
