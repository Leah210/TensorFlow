# coding=utf-8
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# see https://www.tensorflow.org/api_guides/python/reading_data
sess = tf.Session()
file_queue = tf.train.string_input_producer( # string_input_producer creates a FIFO queue for holding the filenames until the reader needs them
    ["iris.csv"]
)
reader = tf.TextLineReader(skip_header_lines=0) # skip_header_lines: An optional int. Defaults to 0. Number of lines to skip from the beginning of every file.
key, value = reader.read(file_queue)
# 可实现
record_defaults = [[0.], [0.], [0.], [0.], [""]] #Default values, in case of empty columns. Also specifies the type of the decoded result.
# 可实现
# record_defaults = [tf.constant([], dtype=tf.float32),
#                    tf.constant([], dtype=tf.float32),
#                    tf.constant([], dtype=tf.float32),
#                    tf.constant([], dtype=tf.float32),
#                    tf.constant([], dtype=tf.string)
#                    ]
col1,col2,col3,col4,col5 = tf.decode_csv(value, record_defaults=record_defaults) #Convert CSV records to tensors. Each column maps to one tensor
features = tf.stack([col1,col2,col3,col4]) #stack里面要放上类型相同的tensor

sess.run(tf.global_variables_initializer())# 教材上是tf.initailze_all_variables(),但是该函数弃用了
sess.run(tf.local_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

# 这里就是按行读取
for i in range(0,5):
    example,label = sess.run([features, col5])
    print(example,label)
coord.request_stop()
coord.join(threads)

