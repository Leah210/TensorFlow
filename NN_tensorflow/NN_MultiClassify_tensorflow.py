# -*- coding: utf-8 -*-
# Time    : 2018/9/6/0006 21:06
# Author  :liangye
# FileName: NN_MultiClassify_tensorflow.py

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn import preprocessing
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

sess = tf.Session()
# read data-------------------------------------------------------------------------------------------------------------
df = pd.read_csv("./wine.csv", header=0)
# print(df.head())
name_fea = df.columns
# print(name_fea)
X = df[name_fea[1:13]].values
# print(x[1:5,:])
y = df['Wine'].values-1  #文件中用wine的值为1,2,3
Y = tf.one_hot(indices=y,depth=3,on_value=1.,off_value=0.,axis=1,name='a').eval(session=sess)

X, Y = shuffle(X,Y)
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X) #标准化

# create model----------------------------------------------------------------------------------------------------------
x = tf.placeholder('float',[None,12]) #None 是不知道有多少个样本
# input2hidden_weights
W = tf.Variable(tf.zeros([12,3]))
# input2hidden_bias
b = tf.Variable(tf.zeros([3]))
# just one hidden_layer with 3 units
y = tf.nn.softmax(tf.matmul(x,W) + b)

# define loss and optimizer---------------------------------------------------------------------------------------------
y_true = tf.placeholder(dtype='float')
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true*tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

# train model-----------------------------------------------------------------------------------------------------------
sess.run(tf.global_variables_initializer())

for i in range(100):
    X,Y = shuffle(X,Y)
    Xtr = X[0:140,:]
    Ytr = Y[0:140,:]
    Xte = Y[140:178,:]
    Yte = Y[140:178,:]
    Xtr, Ytr = shuffle(Xtr, Ytr, random_state=0)
    batch_xs, batch_ys = Xtr, Ytr

    sess.run(train_step, feed_dict={x:batch_xs,y_true:batch_ys})
    cost = sess.run(cross_entropy, feed_dict={x:batch_xs,y_true:batch_ys})

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_true,1))
    acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) # Casts a tensor to a new type.
    print('iters ', i,', the avg acc is = ', sess.run(acc, feed_dict={x:batch_xs,y_true:batch_ys}))