# -*- coding: utf-8 -*-
# Time    : 2018/10/31 16:40
# Author  :liangye
# FileName: layers.py

import tensorflow as tf

# weights in inital network
def weights_variable_initial(shape, stddev=0.1, name = "weight"):
    initial_weights = tf.truncated_normal(shape, stddev)
    return tf.Variable(initial_weights, name)

# weights in deconvolution layer, but all inital weights can call weights_variable_initial
def weights_variable_deconv(shape, stddev = 0.1, name = "weight_deconv"):
    weights = tf.truncated_normal(shape, stddev)
    return tf.Variable(weights, name)

# bias in intital network
def bias_variable(shape, name = "bias"):
    initial_bias = tf.constant(0.1, shape = shape)
    return tf.Variable(initial_bias, name)

# define conv layer
def conv2d(input_x, weight, bias):
    conv = tf.nn.conv2d(input_x, tf.constant(weight), strides = (1, 1, 1, 1), padding = "SAME")
    return tf.nn.bias_add(conv, bias)

# define deconvolutional layer
def deconv2d(x, filter,b, stride):
    x_shape = tf.shape(x)
    # x_shape[0] is batch_size
    output_shape = tf.stack([x_shape[0],x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
    # the shape of w is [height, width, output_channels, in_channels]
    rs = tf.nn.conv2d_transpose(x, filter, output_shape, strides = [1, stride, stride, 1], padding = "SAME", name = "conv2d_transpose", data_format = "NHWC")
    rs  = tf.nn.relu(tf.nn.bias_add(rs, b, name = 'add_bias'), name = 'relu')
    return rs


# define dropout layer
def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob = keep_prob)

# define pooling layer
def pooling_layer(x, n):
    return tf.nn.max_pool(x, ksize = [1,n,n,1], strides = [1, n, n, 1], padding = "SAME", name = "max_pooling")

# define output layer
def output_layer(x, name='sigmoid'):
    result = []
    if name == 'sigmoid':
        result = tf.nn.sigmoid(x)
    if name == 'relu':
        result = tf.nn.relu(x)
    return result

# define concat function
def crop_and_concat(x1,x2):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offset for the top left corner of the crop
    offset = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0] # slice index
    size = [-1, x2_shape[1], x2_shape[2], -1] # use x2 width and height
    x1_crop = tf.slice(x1, offset, size) # slice specific size tensor

    return tf.concat([x1_crop, x2], 3) # concat the x1_crop and x2 by axis 3

# define pixel wise softmax function
def pixel_wise_softmax(output_map):
    # print('the shape of output_map is = ', tf.shape(output_map))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    max_pixel = tf.reduce_max(output_map, axis = 3, keepdims = True) # the max of innerest layer
    print('compute max_pixel done! the shape of max_pixel is = ', sess.run(tf.shape(max_pixel)))
    normalize_exp_map = tf.exp(output_map-max_pixel) # normalize the feature map and compute exponential--element wise
    print("the normalize exp is = ", sess.run(normalize_exp_map))
    print('compute normalize_exp_map done! the shape of normalize_exp_map is = ', sess.run(tf.shape(normalize_exp_map)))
    normalize = tf.reduce_sum(normalize_exp_map, axis = 3, keepdims = True) # compute the sum of the innerest layer values
    print('compute normalize done! the shape of normalize is = ', sess.run(normalize))
    print('the normalize is = ',sess.run(tf.slice(normalize,[0,0,0,0],[1,324,324,1])))
    rs = normalize_exp_map / normalize
    # print('the shape of softmax out_map in fucntion pixel_wise_softmax = ', sess.run(rs))
    return normalize_exp_map

# define the loss function
def cross_entropy(y, output_map):
    new_output_map = tf.clip_by_value(output_map, 1e-10, 1.0)
    return -tf.reduce_mean(y * tf.log(new_output_map), name = "cross_entropy")

# define normalization the layer
def normalization_img(x):
    variance_epsilon = 0.001
    wb_mean, wb_var = tf.nn.moments(x, axes = [0,1,2])
    rs = tf.nn.batch_normalization(x, mean = wb_mean, variance = wb_var, offset = None, scale = None, variance_epsilon = variance_epsilon)
    return rs

