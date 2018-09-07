# -*- coding: utf-8 -*-
# Time    : 2018/9/6/0006 11:25
# Author  :liangye
# FileName: NN_tensorflow.py

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

trainsamples = 200
testsample = 60

# 定义两种模型,检查效果
def model1(X, hidden_weights, hidden_bias, hidden2out_weights, hidden2out_bias):
    hidden_layer = tf.nn.sigmoid(tf.matmul(X, hidden_weights)+ hidden_bias)
    return tf.nn.sigmoid(tf.matmul(hidden_layer, hidden2out_weights) + hidden2out_bias)

def model2(X, hidden_weights, hidden_bias, hidden2out_weights, hidden2out_bias):
    hidden_layer = tf.nn.sigmoid(tf.matmul(X, hidden_weights)+ hidden_bias)
    return tf.matmul(hidden_layer, hidden2out_weights)

dsX = np.linspace(-1, 1, trainsamples+testsample).transpose()
dsY = 0.4*pow(dsX,2) + 2*dsX + np.random.randn(*dsX.shape) * 0.22 + 0.8
# plt.scatter(dsX, dsY)
# plt.show()
# print('the shape of dsX = ', tf.Session().run(tf.shape(dsX)))

X = tf.placeholder(dtype='float')
Y = tf.placeholder(dtype='float')
# 创建输入层到隐藏层的权重
in2h_weight = tf.Variable(tf.random_normal([1,10],stddev=0.1))
# 创建隐藏层到输出层的权重
h2out_weight = tf.Variable(tf.random_normal([10,1],stddev=0.1))
# 创建输入层到隐藏层的偏差
in2h_bias = tf.Variable(tf.random_normal([1,10],stddev=0.0))
# 创建输入层到隐藏层的偏差
h2o_bias = tf.Variable(tf.random_normal([10,1],stddev=0.0))
# 创建模型的工作流
# model_y = model1(X, in2h_weight, in2h_bias, h2out_weight, h2o_bias)
model_y = model2(X, in2h_weight, in2h_bias, h2out_weight, h2o_bias)
# 计算损失
cost = tf.pow(model_y-Y, 2)/2
# 优化
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 迭代100次
    for i in range(1,100):
        dsX, dsY = shuffle(dsX.transpose(),dsY) # 打乱数据
        train_x, train_y = dsX[0:trainsamples], dsY[0:trainsamples]
        # 训练模型
        for (x1,y1) in zip(train_x, train_y):
            # 这里[[x1]]的原因: 定义权重和偏差的时候, 是根据单独样本来计算的, 因此这里需要提取出样本的具体值, 也需要遍历每个样本, 而不是利用python中brodcast功能
            sess.run(train_op, feed_dict={X:[[x1]], Y:y1})
        test_x, test_y = dsX[trainsamples:(trainsamples+testsample)], dsY[trainsamples:(trainsamples+testsample)]
        cost1 = 0. # 命名不要覆盖上面的工作流
        for x1,y1 in zip(test_x, test_y):
            cost1+=sess.run(cost, feed_dict={X:[[x1]], Y:y1})/testsample
        if i%10==0:
            print("avg cost for epoch ", str(i), ":", cost1)