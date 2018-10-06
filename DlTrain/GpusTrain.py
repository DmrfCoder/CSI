import os
import time

import tensorflow as tf

import numpy as np
from tensorflow.contrib.timeseries.python.timeseries import model

from DlTrain.CNN import CNN
from DlTrain.LSTM import LSTM
from DlTrain.Parameters import lstmInputDimension, tfRootPath, logRoot, pbRoot, matrixRoot, trainBatchSize, \
    lstmTimeStep, trainingIterations, valBatchSize, valPerTrainIterations
from Util.Matrix import drawMatrix
from Util.ReadAndDecodeUtil import read_and_decode


os.environ['CUDA_VISIBLE_DEVICES']='0,1'


def average_losses(loss):
    tf.add_to_collection('losses', loss)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses')

    # Calculate the total loss for the current tower.
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n(losses + regularization_losses, name='total_loss')

    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
    return total_loss


def average_gradients(tower_grads):
    average_grads = []

    for i in range(2, 12):
        del tower_grads[1][2]

    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = [g for g, _ in grad_and_vars]
        # Average over the 'tower' dimension.

        grad = tf.stack(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def feed_all_gpu(inp_dict, models, payload_per_gpu, batch_x, batch_y):
    for i in range(len(models)):
        x, y, _, _, _ = models[i]
        start_pos = i * payload_per_gpu
        stop_pos = (i + 1) * payload_per_gpu
        inp_dict[x] = batch_x[start_pos:stop_pos]
        inp_dict[y] = batch_y[start_pos:stop_pos]
    return inp_dict


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

        trainLogPath = logRoot + '/' + rootType + '/' + which + '/0/train'
        valLogPath = logRoot + '/' + rootType + '/' + which + '/0/val'

    else:

        intLastIndex = logsort
        intLastIndex += 1

        lastIndex = str(intLastIndex)
        os.mkdir(logRoot + '/' + rootType + '/' + which + '/' + lastIndex)
        os.mkdir(logRoot + '/' + rootType + '/' + which + '/' + lastIndex + '/train')
        os.mkdir(logRoot + '/' + rootType + '/' + which + '/' + lastIndex + '/val')
        trainLogPath = logRoot + '/' + rootType + '/' + which + '/' + lastIndex + '/train'
        valLogPath = logRoot + '/' + rootType + '/' + which + '/' + lastIndex + '/val'

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


def openLabel(label, batch_size):
    l = np.zeros(shape=(batch_size, 5))
    for li in range(batch_size):
        l[li][label[li]] = 1

    return l


def multi_gpu(baseIr, rootType, which, InputDimension=lstmInputDimension, num_gpu=2):
    print 'start...'
    folders_dict = init_folder(rootType=rootType, which=which)

    train_tf_path = tfRootPath + rootType + '/' + which + '/train.tfrecords'
    val_tf_path = tfRootPath + rootType + '/' + which + '/val.tfrecords'

    batch_size = trainBatchSize * num_gpu

    tf.reset_default_graph()

    with tf.Session() as sess:
        with tf.device('/cpu:0'):

            x_train, y_train = read_and_decode(train_tf_path)
            num_threads = 5
            min_after_dequeue_train = 10000

            train_capacity_train = min_after_dequeue_train + num_threads * trainBatchSize
            train_x_batch, train_y_batch = tf.train.shuffle_batch([x_train, y_train],
                                                                  batch_size=batch_size, capacity=train_capacity_train,
                                                                  min_after_dequeue=min_after_dequeue_train)

            x_val, y_val = read_and_decode(val_tf_path)
            num_threads = 5
            min_after_dequeue_val = 2500
            train_capacity_val = min_after_dequeue_val + num_threads * trainBatchSize
            val_x_batch, val_y_batch = tf.train.shuffle_batch([x_val, y_val],
                                                              batch_size=batch_size, capacity=train_capacity_val,
                                                              min_after_dequeue=min_after_dequeue_val)

            init = tf.global_variables_initializer()
            sess.run(init)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            #
            # for thread in threads:
            #     try:
            #
            #         thread.start()
            #         print 'start thread'
            #     except RuntimeError:
            #         print 'start thread exception'
            #         break

            merged = tf.summary.merge_all()

            trainPredictionFile = open(folders_dict['trainPredictionTxtPath'], 'wb')
            trainReallyTxtFile = open(folders_dict['trainReallyTxtPath'], 'wb')

            trainLogWriter = tf.summary.FileWriter(folders_dict['trainLogPath'], sess.graph)
            valLogWriter = tf.summary.FileWriter(folders_dict['valLogPath'], sess.graph)

            with tf.name_scope('learning_rate'):
                learning_rate = tf.placeholder(tf.float32, shape=[])
                tf.summary.scalar('learning_rate', learning_rate)

            opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

            print('build model...')
            print('build model on gpu tower...')
            models = []
            for gpu_id in range(num_gpu):
                with tf.device('/gpu:%d' % gpu_id):
                    print('tower:%d...' % gpu_id)

                    with tf.name_scope('tower_%d' % gpu_id):
                        with tf.variable_scope('cpu_variables', reuse=gpu_id > 0):
                            x = tf.placeholder(tf.float32, shape=[None, lstmTimeStep * InputDimension],
                                               name='inputLstm')
                            y = tf.placeholder(tf.int32, shape=[None, 5], name='Label')

                            cnnInput = LSTM(x)
                            pred = CNN(cnnInput)
                            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y)
                            loss = tf.reduce_mean(loss)

                            grads = []
                            grads = opt.compute_gradients(loss)

                            models.append((x, y, pred, loss, grads))

            print('build model on gpu tower done.')

            print('reduce model on cpu...')
            tower_x, tower_y, tower_preds, tower_losses, tower_grads = zip(*models)

            with tf.name_scope('Loss'):
                aver_loss_op = tf.reduce_mean(tower_losses)
                tf.summary.scalar('Loss', aver_loss_op)

            apply_gradient_op = opt.apply_gradients(average_gradients(tower_grads))

            all_y = tf.reshape(tf.stack(tower_y, 0), [-1, 5], 'all_y')
            all_pred = tf.reshape(tf.stack(tower_preds, 0), [-1, 5], 'all_pred')
            re_y = tf.argmax(all_y, 1, name='re_y')
            pr_y = tf.argmax(all_pred, 1, name='pr_y')

            correct_pred = tf.equal(tf.argmax(all_y, 1), tf.argmax(all_pred, 1), 'correct_pred')

            with tf.name_scope('Accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_pred, 'float'))
                tf.summary.scalar('Accuracy', accuracy)

            print('reduce model on cpu done.')

            print('run train op...')
            sess.run(tf.global_variables_initializer())

            start_time = time.time()
            payload_per_gpu = trainBatchSize
            total_batch = int(trainingIterations/num_gpu)
            avg_loss = 0.0

            inp_dict = {}
            inp_dict[learning_rate] = baseIr

            for batch_idx in range(total_batch):
                batch_x, batch_y = sess.run([train_x_batch, train_y_batch])
                batch_x = np.reshape(batch_x, newshape=(-1, 72000))
                # batch_y=np.reshape(batch_y,newshape=(-1,1))
                batch_y = openLabel(batch_y, batch_size)
                inp_dict = feed_all_gpu(inp_dict, models, payload_per_gpu, batch_x, batch_y)
                _, _loss = sess.run([apply_gradient_op, aver_loss_op], inp_dict)

                # trainLogWriter.add_summary(_merged, batch_idx)

                print('step: %d ,Train loss:%.4f' % (batch_idx, _loss))
                avg_loss += _loss

                if batch_idx % valPerTrainIterations == 0:
                    batch_x, batch_y = sess.run([val_x_batch, val_y_batch])
                    batch_x = np.reshape(batch_x, newshape=(-1, 72000))
                    batch_y = openLabel(batch_y, batch_size)
                    inp_dict_val = feed_all_gpu({}, models, payload_per_gpu, batch_x, batch_y)
                    batch_pred, batch_y = sess.run([all_pred, all_y], inp_dict_val)

                    val_accuracy, _mergedval, reall_y, predic_y = sess.run([accuracy, merged, re_y, pr_y],
                                                                           {all_y: batch_y, all_pred: batch_pred})
                    print('Val Accuracy:                 %0.4f%%' % (100.0 * val_accuracy))
                    np.savetxt(trainReallyTxtFile, reall_y)
                    np.savetxt(trainPredictionFile, predic_y)

                    valLogWriter.add_summary(_mergedval, batch_idx)

                if batch_idx==10:
                    break


            trainLogWriter.close()
            valLogWriter.close()

            trainReallyTxtFile.close()
            trainPredictionFile.close()


            constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["correct_pred"])
            with tf.gfile.FastGFile(folders_dict['pbPath'], mode='wb') as f:
                f.write(constant_graph.SerializeToString())

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=5)

        sess.close()

        drawMatrix(folders_dict['trainReallyTxtPath'], folders_dict['trainPredictionTxtPath'],
                   folders_dict['matrixPicturePath'])

        stop_time = time.time()
        elapsed_time = stop_time - start_time
        print('Cost time: ' + str(elapsed_time) + ' sec.')
        print('training done.\n')


rootType = ['AmplitudeWithout_PhaseWith', 'AmplitudeWithOut_PhaseWithout', 'AmplitudeWith_PhaseWith',
            'AmplitudeWith_PhaseWithout', 'OnlyAmplitude', 'OnlyPhase']
for i in range(6):
    print str(i)+'....'
    if i < 5:
        multi_gpu(rootType=rootType[i], which='fixed', baseIr=0.2)
        multi_gpu(rootType=rootType[i], which='open', baseIr=0.15)
        multi_gpu(rootType=rootType[i], which='semi', baseIr=0.1)
    else:
        multi_gpu(rootType=rootType[i], which='fixed', baseIr=0.2, InputDimension=180)
        multi_gpu(rootType=rootType[i], which='open', baseIr=0.15, InputDimension=180)
        multi_gpu(rootType=rootType[i], which='semi', baseIr=0.1, InputDimension=180)
