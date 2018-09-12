# -*- coding: utf-8 -*-
# Time    : 2018/9/7/0007 14:48
# Author  :liangye
# FileName: CNN_conv.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
sess = tf.Session()

file_name = tf.train.string_input_producer(['04.gif']) # note: the input is a 1-D tensor
reader = tf.WholeFileReader()
key, value = reader.read(file_name)
image = tf.image.decode_gif(value) #由于jpg得到的维度是[640 640   3], 而conv2d需要的是[* * * *], 因此这里增加一个维度
# image = tf.constant([1 image])

# define the kernel param
kernel=tf.constant(
[
 [[[0.]],[[0.]],[[0.]]],
 [[[0.]],[[1.]],[[0.]]],
 [[[0.]],[[0.]],[[0.]]]
]
)
sess.run(tf.global_variables_initializer())
print('the shape of kernel = ', sess.run(tf.shape(kernel)))

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord,sess=sess)
print('the shape of image = ', sess.run(tf.shape(image))) # image can be readed after starting the queue runners

# get the first channel(gif consists of many images with 3 channels)
image_tensor = tf.image.rgb_to_grayscale(sess.run([image])[0])
# select the first image with first channel
image_tensor = tf.slice(image_tensor,[0,0,0,0],[1,200,200,1])
print('the shape of image_tensor = ', sess.run(tf.shape(image_tensor)))

# [batch, in_height, in_width, in_channels]
# now, iamge_tensor has 31 batch,200*200 pixel,1 channel
image_tensor_conv = tf.nn.conv2d(tf.cast(image_tensor, tf.float32), kernel, [1,1,1,1],'SAME')
print('the shape of image_tensor_conv = ', sess.run(tf.shape(image_tensor_conv)))

# the final result was affected by the kernel
file = open('blur_cat.jpg','wb+')
# encode_jpeg use 3-D uint8 Tensor of shape `[height, width, channels], so herer we select the first sample for reshape the image_tensor_conv
blur_cat = tf.image.encode_jpeg(tf.reshape( tf.cast(image_tensor_conv[0],tf.uint8), tf.shape(image_tensor_conv[0])))
blur_cat_run = sess.run(blur_cat)
file.write(blur_cat.eval(session=sess))
file.close()

coord.request_stop()
coord.join(threads)





