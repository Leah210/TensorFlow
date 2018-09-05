# coding=utf-8
import tensorflow as tf
sess = tf.Session()
filename_queue = tf.train.string_input_producer(['cat.jpg'])
reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)
image = tf.image.decode_jpeg(value)
flipImageUpDown = tf.image.encode_jpeg(tf.image.flip_up_down(image)) #上下翻转并编码
flipImageLeftRight = tf.image.encode_jpeg(tf.image.flip_left_right(image)) # 左右翻转并编码

sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord,sess=sess)
exam1 = sess.run(flipImageUpDown)
exam2 = sess.run(flipImageLeftRight)

# print(exam1)
file = open('flippedUpDown.jpg', 'wb+')
file.write(flipImageUpDown.eval(session=sess)) # 写文件
file.close()
file = open('flippedLeftRight.jpg', 'wb+')
file.write(flipImageLeftRight.eval(session=sess)) # 写文件
file.close()