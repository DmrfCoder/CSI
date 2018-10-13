# -*-coding:utf-8-*-
import os
import threading

import tensorflow as tf
import numpy as np

from DlTrain.CNN import CNN
from DlTrain.Data import Data
from DlTrain.LSTM import LSTM

from DlTrain.Parameters import lstmTimeStep, lstmInputDimension, valIterations, \
    trainBatchSize, trainingIterations, valBatchSize, \
    valPerTrainIterations, logRoot, pbRoot, matrixRoot, tfRootPath, init_folder

from Util.Matrix import drawMatrix
from Util.ReadAndDecodeUtil import read_and_decode
os.environ['CUDA_VISIBLE_DEVICES']='0,1'

def trainauto(baseIr, rootType, which, InputDimension=lstmInputDimension, gpu_code=0):
    tf.reset_default_graph()
    with tf.Session() as sess:

        train_tf_path = tfRootPath + rootType + '/' + which + '/train.tfrecords'
        val_tf_path = tfRootPath + rootType + '/' + which + '/val.tfrecords'
        print 'start...'
        folders_dict = init_folder(rootType=rootType, which=which)

        lstmInput = tf.placeholder(tf.float32, shape=[None, lstmTimeStep * InputDimension], name='inputLstm')
        Label = tf.placeholder(tf.int32, shape=[None, ], name='Label')

        cnnInput = LSTM(lstmInput,InputDimension)
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

        x_train, y_train = read_and_decode(train_tf_path,InputDimension)
        num_threads = 3
        min_after_dequeue_train = 10000

        train_capacity_train = min_after_dequeue_train + num_threads * trainBatchSize

        # 使用shuffle_batch可以随机打乱输入
        train_x_batch, train_y_batch = tf.train.shuffle_batch([x_train, y_train],
                                                              batch_size=trainBatchSize,
                                                              capacity=train_capacity_train,
                                                              min_after_dequeue=min_after_dequeue_train)

        x_val, y_val = read_and_decode(val_tf_path,InputDimension)
        num_threads = 3
        min_after_dequeue_val = 2500

        train_capacity_val = min_after_dequeue_val + num_threads * trainBatchSize

        # 使用shuffle_batch可以随机打乱输入
        val_x_batch, val_y_batch = tf.train.shuffle_batch([x_val, y_val],
                                                          batch_size=valBatchSize, capacity=train_capacity_val,
                                                          min_after_dequeue=min_after_dequeue_val)

        coord = tf.train.Coordinator()

        isTestMode = False  # 是否是验证阶段
        isTestCode = False  # 是否是测试代码模式（产生随机数据）

        isWriteFlag = True  # 是否将label写入文件
        saver = tf.train.Saver(max_to_keep=1)
        merged = tf.summary.merge_all()

        trainLogPath = folders_dict['trainLogPath']
        valLogPath = folders_dict['valLogPath']
        trainPredictionTxtPath = folders_dict['trainPredictionTxtPath']
        trainReallyTxtPath = folders_dict['trainReallyTxtPath']
        pbPath = folders_dict['pbPath']
        matrixPicturePath = folders_dict['matrixPicturePath']

        sess.run(tf.global_variables_initializer())
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if not isTestMode:

            trainLogWriter = tf.summary.FileWriter(trainLogPath, sess.graph)
            valLogWriter = tf.summary.FileWriter(valLogPath, sess.graph)

            if isWriteFlag:
                valPredictionTxtFile = open(trainPredictionTxtPath, 'wb')
                valReallyTxtFile = open(trainReallyTxtPath, 'wb')

            for step in range(trainingIterations + 1):

                # X, Y = trainData.getNextManualShuffleBatch(trainBatchSize)
                X, Y = sess.run([train_x_batch, train_y_batch])
                X = np.reshape(X, newshape=(-1, 200*InputDimension))

                sess.run(trainOp, feed_dict={lstmInput: X, Label: Y})
                if step % valPerTrainIterations == 0:
                    valX, valY = sess.run([val_x_batch, val_y_batch])
                    valX = np.reshape(valX, newshape=(-1, 200*InputDimension))

                    out_labels = sess.run(predictionLabels, feed_dict={lstmInput: valX, Label: valY})

                    if isWriteFlag:
                        np.savetxt(valReallyTxtFile, valY)
                        np.savetxt(valPredictionTxtFile, out_labels)

                    valLoss, valAccuracy = sess.run([loss, Accuracy], feed_dict={lstmInput: valX, Label: valY})
                    print('step:%d, valLoss:%f, valAccuracy:%f' % (step, valLoss, valAccuracy))
                    valSummary, _ = sess.run([merged, trainOp], feed_dict={lstmInput: X, Label: Y})
                    valLogWriter.add_summary(valSummary, step)


                trainSummary, _ = sess.run([merged, trainOp], feed_dict={lstmInput: X, Label: Y})

                trainLogWriter.add_summary(trainSummary, step)

            constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                          ["PredictionLabels"])

            if not isTestCode:
                trainLogWriter.close()
                valLogWriter.close()

            if isWriteFlag:
                valPredictionTxtFile.close()
                valReallyTxtFile.close()

            with tf.gfile.FastGFile(pbPath, mode='wb') as f:
                f.write(constant_graph.SerializeToString())

            drawMatrix(reallyTxtPath=trainReallyTxtPath, predictionTxtPath=trainPredictionTxtPath,
                       matrixPath=matrixPicturePath)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=5)
            sess.close()


rootType = ['AmplitudeWithout_PhaseWith', 'AmplitudeWithOut_PhaseWithout', 'AmplitudeWith_PhaseWith',
            'AmplitudeWith_PhaseWithout', 'OnlyAmplitude', 'OnlyPhase']


#
#
# trainauto(rootType=rootType[3], which='fixed', baseIr=0.2, gpu_code=0)
# trainauto(rootType=rootType[3], which='open', baseIr=0.15, gpu_code=0)
# trainauto(rootType=rootType[3], which='semi', baseIr=0.1, gpu_code=1)

trainauto(rootType=rootType[4], which='fixed', baseIr=0.2, InputDimension=180, gpu_code=1)
trainauto(rootType=rootType[4], which='open', baseIr=0.15, InputDimension=180, gpu_code=1)
trainauto(rootType=rootType[4], which='semi', baseIr=0.1, InputDimension=180, gpu_code=1)

trainauto(rootType=rootType[5], which='fixed', baseIr=0.2, InputDimension=180, gpu_code=1)
trainauto(rootType=rootType[5], which='open', baseIr=0.15, InputDimension=180, gpu_code=1)
trainauto(rootType=rootType[5], which='semi', baseIr=0.1, InputDimension=180, gpu_code=1)




#
# for i in range(6):
#     print str(i) + '....'
#     if i < 5:
#         trainauto(rootType=rootType[i], which='fixed', baseIr=0.2, gpu_code=0)
#         trainauto(rootType=rootType[i], which='open', baseIr=0.15, gpu_code=1)
#         trainauto(rootType=rootType[i], which='semi', baseIr=0.1)
#     else:
#         trainauto(rootType=rootType[i], which='fixed', baseIr=0.2, InputDimension=180)
#         trainauto(rootType=rootType[i], which='open', baseIr=0.15, InputDimension=180)
#         trainauto(rootType=rootType[i], which='semi', baseIr=0.1, InputDimension=180)
