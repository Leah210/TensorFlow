# -*- coding: utf-8 -*-
# Time    : 2018/09/26 11:59
# Author  :liangye
# FileName: vgg.py

import numpy as np
import tensorflow as tf
import scipy.io
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 定义网络每层类型
VGG19_LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4'
)

def load_net(data_path):
    '''
    加载已经训练好的参数数据
    :param data_path:
    :return:
    '''
    # mat = data["key_name"][0]之后, 才开始访问
    # 如果mat中是struct,需要加入[0][0]才能开始访问
    # 如果mat中是list,直接访问
    # 先要研究好mat中的数据存储的形式,才能够正确访问
    data = scipy.io.loadmat(data_path)
    if not all(i in data for i in ('layers', 'classes', 'normalization')):
        raise ValueError(
            "You're using the wrong VGG19 data. Please follow the instructions in the README to download the correct data.")
    mean = data['normalization'][0][0][0] # 得到的是image, 224*224*3
    mean_pixel = np.mean(mean, axis=(0,1)) # 每个通道求均值, 最终得到1*3的结果
    weights = data['layers'][0]
    return  weights, mean_pixel  # 注意返回的顺序

def conv_layers(input_image, weights, bias):
    '''
    卷积操作
    :param input_image:
    :param weights:
    :param bias:
    :return:
    '''
    conv = tf.nn.conv2d(input_image, tf.constant(weights), strides=(1,1,1,1), padding="SAME")
    return tf.nn.bias_add(conv,bias)

def pooling_layers(input_image,pooling):
    if pooling == "avg":
        return tf.nn.avg_pool(input_image,ksize=(1,2,2,1),strides=(1,2,2,1),padding='SAME')
    else:
        return tf.nn.max_pool(input_image, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

def net_preload(weights, input_image, pooling):
    '''
    初始参数加载后,直接对input做一次计算
    :param weights:
    :param input_image:
    :param pooling:
    :return:
    '''
    net = {}  # 定义字典
    current = input_image
    for i, name in enumerate(VGG19_LAYERS):
        kind = name[0:4]  # 提取第1个字符到第4个字符,辨别当前是卷积还是池化层,因为mat中也是根据该顺序进行存储
        if kind == 'conv':
            # print(kind)
            # print(weights)
            kernels, bias = weights[i][0][0][0][0]  # 第1-2个0是: 选择了第i个元素后,是struct,需要用[0][0]进入. 第3个0是: 进入后选择第1个元素(该元素是list,因此可以直接访问), 第4个0是: 选择第1个元素
            # 在mat中,权重的存储方式为[width, height, in_channels, out_channels]
            # 在TensorFlow中,权重的存储方式为[height, width, in_channels, out_channels]
            # 因此要对kernels进行转换
            kernels = np.transpose(kernels,axes=(1,0,2,3))
            bias = np.reshape(bias,-1)
            current = conv_layers(current,kernels,bias)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = pooling_layers(current,pooling)
        net[name] = current # 将此次循环的结果(卷积或者激活函数或者池化的操作)存入net中
    assert len(net) == len(VGG19_LAYERS)
    return net

# VGG-19需要对输入图片进行一步预处理，把每个像素点的取值减去训练集算出来的RGB均值
def preprocess(image,mean_pixel):
    return image-mean_pixel

def unprocess(image,mean_pixel):
    return image+mean_pixel