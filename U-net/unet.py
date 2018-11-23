# -*- coding: utf-8 -*-
# Time    : 2018/11/01 10:53
# Author  :liangye
# FileName: unet.py

import tensorflow as tf
import numpy as np
from layers import *
import scipy.misc as sm
import os
from sys import stderr
from generate_img import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# define the unet
# dropout can be manual added in somewhere
UNET_LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'pool3',
    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'pool4',
    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'dropout5',
    # up_conv6_1 is copy the output of dropout5, and merge the pool4 which has been cropped
    # note: in the layer6, deconv the dropout5 as result1, at the same time crop the output of pool4 as result2, then merge(result1,result2)
    'Ucov1',
    'conv6_1', 'relu6_1', 'conv6_2', 'relu6_2',
    'Ucov2',
    'conv7_1', 'relu7_1', 'conv7_2', 'relu7_2',
    'Ucov3',
    'conv8_1', 'relu8_1', 'conv8_2', 'relu8_2',
    'Ucov4',
    'conv9_1', 'relu9_1', 'conv9_2', 'relu9_2',

    'out_10' # the last number just a flag
)
INPUT_IMAGE_SHAPE = 572
INPUT_IMG_CHANNEL = 1
CLASS_NUM = 2


# define the parameters   [height, width, in_channel, out_channel]
weights = {'conv1_1': weights_variable_initial([3, 3, INPUT_IMG_CHANNEL, 64],name = 'w1_1'),
           'conv1_2': weights_variable_initial([3, 3, 64, 64], name = 'w1_2'),
           'conv2_1': weights_variable_initial([3, 3, 64, 128], name = 'w2_1'),
           'conv2_2': weights_variable_initial([3, 3, 128, 128], name = 'w2_2'),
           'conv3_1': weights_variable_initial([3, 3, 128, 256], name = 'w3_1'),
           'conv3_2': weights_variable_initial([3, 3, 256, 256], name = 'w3_2'),
           'conv4_1': weights_variable_initial([3, 3, 256, 512], name = 'w4_1'),
           'conv4_2': weights_variable_initial([3, 3, 512, 512], name = 'w4_2'),
           'conv5_1': weights_variable_initial([3, 3, 512, 1024], name = 'w5_1'),
           'conv5_2': weights_variable_initial([3, 3, 1024, 1024], name = 'w5_2'),
           'conv6_1': weights_variable_initial([3, 3, 1024, 512], name = 'w6_1'),
           'conv6_2': weights_variable_initial([3, 3, 512, 512], name = 'w6_2'),
           'conv7_1': weights_variable_initial([3, 3, 512, 256], name = 'w7_1'),
           'conv7_2': weights_variable_initial([3, 3, 256, 256], name = 'w7_2'),
           'conv8_1': weights_variable_initial([3, 3, 256, 128], name = 'w8_1'),
           'conv8_2': weights_variable_initial([3, 3, 128, 128], name = 'w8_2'),
           'conv9_1': weights_variable_initial([3, 3, 128, 64], name = 'w9_1'),
           'conv9_2': weights_variable_initial([3, 3, 64, 64], name = 'w9_2'),
           'conv10' : weights_variable_initial([1,1,64, CLASS_NUM], name = 'w_out')
           }

# the shape of w is [height, width, output_channels, in_channels]
decov_weights = {'Ucov1': weights_variable_deconv([2, 2, 512, 1024], name = 'deconv_up1'),
                 'Ucov2': weights_variable_deconv([2, 2, 256, 512], name = 'deconv_up2'),
                 'Ucov3': weights_variable_deconv([2, 2, 128, 256], name = 'deconv_up3'),
                 'Ucov4': weights_variable_deconv([2, 2, 64, 128], name = 'deconv_up4')
                 }

biases = {'bias1_1': bias_variable([64], name = 'bias1_1'),
          'bias1_2': bias_variable([64], name = 'bias1_2'),
          'bias2_1': bias_variable([128], name = 'bias2_1'),
          'bias2_2': bias_variable([128], name = 'bias2_2'),
          'bias3_1': bias_variable([256], name = 'bias3_1'),
          'bias3_2': bias_variable([256], name = 'bias3_2'),
          'bias4_1': bias_variable([512], name = 'bias4_1'),
          'bias4_2': bias_variable([512], name = 'bias4_2'),
          'bias5_1': bias_variable([1024], name = 'bias5_1'),
          'bias5_2': bias_variable([1024], name = 'bias5_2'),
          'bias6_1': bias_variable([512], name = 'bias6_2'),
          'bias6_2': bias_variable([512], name = 'bias6_2'),
          'bias7_1': bias_variable([256], name = 'bias7_2'),
          'bias7_2': bias_variable([256], name = 'bias7_2'),
          'bias8_1': bias_variable([128], name = 'bias8_2'),
          'bias8_2': bias_variable([128], name = 'bias8_2'),
          'bias9_1': bias_variable([64],  name = 'bias9_2'), 
          'bias9_2': bias_variable([64], name = 'bias9_2'),
          'bias10': bias_variable([CLASS_NUM], name = 'bias_out')
        }

up_bias = {
    'Ucov1': weights_variable_deconv([512], name = 'bias_up1'),
    'Ucov2': weights_variable_deconv([256], name = 'bias_up2'),
    'Ucov3': weights_variable_deconv([128], name = 'bias_up3'),
    'Ucov4': weights_variable_deconv([64], name = 'bias_up4')
}



def unet(input_image, weights = weights, biases = biases, keep_prob = 0.80):
    current = input_image
    store_map = []
    store_ind = 3
    output_map= []
    # print('begin the train model~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    for i,name in enumerate(UNET_LAYERS):
        kind, index = name[0:4], name[4:]
        store_middle_ind, store_last_ind = name[4], name[-1]
        weight_name, bias_name = 'conv' + index, 'bias' + index
        if kind == 'conv':
            current = normalization_img(conv2d(current, weight = weights[weight_name], bias = biases[bias_name]))

        if kind == 'Ucov':
            # print('the shape of current in the UP conv is = ', sess.run(tf.shape(current)))
            crop_and_copy = store_map[store_ind]
            # print('the shape of store_map in the UP conv is = ', sess.run(tf.shape(crop_and_copy)))
            deconv_ = deconv2d(current, decov_weights[name], up_bias[name], 2) # need to copy
            # print('the shape of deconv_ is = ', sess.run(tf.shape(deconv_)))
            current = crop_and_copy_concat(crop_and_copy, deconv_)
            # print('after deconv, crop, copy, the shape of current is = ', sess.run(tf.shape(current)))
            store_ind -= 1

        if kind == 'relu':
            current = normalization_img(output_layer(x = current, name = "relu"))
            # store the down step
            if int(store_middle_ind)<5 and int(store_last_ind)==2: # the layer5 needn't to store
                store_map.append(current)

        if kind == 'pool':
            current = pooling_layer(current, 2)
            # print('the shape of current after max_pool is = ', sess.run(tf.shape(current)))

        if kind == 'drop':
            current = dropout(current, keep_prob = keep_prob)

        if kind == 'out_':
            # note: must be flattened using 1*1 conv
            current = conv2d(current, weight = weights[weight_name], bias = biases[bias_name])
            output_map = output_layer(current, name = 'sigmoid')
            output_map = tf.reduce_mean(output_map, axis = 3, keepdims = True)
            # output_map = pixel_wise_softmax(output_map) # output_map also is a prediction
    # out_shape = output_map.get_shape().as_list()
    # level_tensor = tf.constant(0.6, dtype = tf.float32, shape = out_shape)
    # output_map = tf.cast(tf.greater(output_map, level_tensor),tf.float32)
    # print('end the train model~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    return output_map

def compute_loss(feature_map, label):
    feature_map = tf.reshape(feature_map,[-1, CLASS_NUM])
    label = tf.reshape(label, [-1, CLASS_NUM])
    # loss and optimizer
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = feature_map, labels = label)
    loss = tf.reduce_mean(loss)
    # print('end compute the loss~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    return  loss


# define the copy and crop and merge function
def crop_and_copy_concat(need_crop, need_copy):
    need_copy_tensor_shape = tf.shape(need_copy)
    need_crop_tensor_shape = tf.shape(need_crop)
    crop_tensor = tf.slice(input_ = need_crop,
                           begin = [0,
                                    (need_crop_tensor_shape[1] - need_copy_tensor_shape[1]) // 2,
                                    (need_crop_tensor_shape[2] - need_copy_tensor_shape[2]) // 2,
                                    0],
                           size = [-1,
                                   need_copy_tensor_shape[1],
                                   need_copy_tensor_shape[2],
                                   -1
                                   ],
                           name = "crop_tensor")
    result_tensor = tf.concat([crop_tensor, need_copy], axis = 3)  # concat two tensor with the last axis
    return result_tensor


# define accuracy
def accuracy(prediction, input_label):
    out_shape = prediction.get_shape().as_list()
    level_tensor = tf.constant(0.5, dtype = tf.float32, shape = out_shape)
    prediction = tf.cast(tf.greater(prediction, level_tensor),tf.float32)
    correct_prediction = tf.cast(tf.equal(prediction, input_label),tf.float32)
    acc = tf.reduce_mean(correct_prediction)
    return correct_prediction, acc

# save model
def save_model(sess, save_path, epoch):
    saver = tf.train.Saver()
    saver.save(sess, save_path, global_step = epoch)