# -*- coding: utf-8 -*-
# @Time    : 2018/9/6/0006 10:51
# @Author  :liangye
# @FileName: LR_tensorflow.py
# @Software: PyCharm
# coding=utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import pandas as pd

df = pd.read_csv('H:/PycharmProjects/LogisticRegression/data/CHD.csv', header=0)

# param
learning_rate = 0.2
training_epochs = 5
batch_size = 100
display_step = 1
sess = tf.Session()
b = np.zeros((100,2))

x = tf.placeholder('float',[None,1])
y = tf.placeholder('float',[None,2])
W = tf.Variable(tf.zeros([1,2]))
b = tf.Variable(tf.zeros([2]))

# model
activation = tf.nn.softmax(tf.multiply(x,W) + b)

# cost
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(activation),reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session():
    sess.run(init)
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = 400/batch_size
        for i in range(int(total_batch)):
            temp = tf.one_hot(indices=df['chd'].values, depth=2, on_value=1, off_value=0, axis=-1, name="a")
            batch_xs, batch_ys = (np.transpose([df['age']])-44.38)/11.721327, temp
            sess.run(optimizer, feed_dict={x: batch_xs.astype(float),y:batch_ys.eval()})
            avg_cost+=sess.run(cost, feed_dict={x:batch_xs.astype(float),y:batch_ys.eval()})/total_batch
        if epoch%display_step==0:
            print("epoch:",  '%05d' % (epoch+1), "cost=", "{:.8f}".format(avg_cost))
            # trX = np.linspace(-30,30,100)
            # print(b.eval())
            # print(W.eval())
            # Wdos = 2*W.eval()[0][0]/11.721327
            # bdos = 2*b.eval()[0]
            # trY = np.exp()