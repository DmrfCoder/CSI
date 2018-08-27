from Util.ReadAndDecodeUtil import read_and_decode
import tensorflow as tf

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串类型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def split(data,len,target_path):
    writer = tf.python_io.TFRecordWriter(target_path)
    data_len=len(data)
    x,y= tf.train.batch([data[0], data[1]], batch_size=len)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        threads = tf.train.start_queue_runners(sess=sess)

        train_x, train_y = sess.run([x, y])
        a=train_y[0]
        for i in range(1,len):
            if train_y[i]!=a:
                a=train_y[i]
                break
        if a==train_y[0]:
            data_raw = train_x
            label =a
            data_bytes = data_raw.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(label),
                'data_raw': _bytes_feature(data_bytes)
            }))
            print('doing:' + str(i) + ' label:' + str(label))
            writer.write(example.SerializeToString())

    writer.close()

if __name__=='__main__':
    semi_path = 'E:\\yczhao Data\\semi.tfrecords'
    semi_target_path = 'E:\\yczhao Data\\semi_200.tfrecords'
    data=read_and_decode(semi_target_path)
    split(data,200,semi_target_path)