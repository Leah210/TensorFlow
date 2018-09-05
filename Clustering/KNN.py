# coding=utf-8
import tensorflow as tf
import numpy as np
import sklearn as sk
import time
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_circles

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

N = 210
K = 2
MAX_ITERS = 1000
cut = int(N*0.7)

start = time.time()
data, features = make_circles(n_samples=N,shuffle=True,noise=0.12,factor=0.4)
tr_data, tr_features = data[:cut], features[:cut]
te_data, te_feature = data[cut:], features[cut:]

fig, ax = plt.subplots()
ax.scatter(tr_data.transpose()[0], tr_data.transpose()[1], marker = 'o', s = 100, c = tr_features, cmap=plt.cm.coolwarm )
plt.show()

points = tf.Variable(data)
cluster_assignment = tf.Variable(tf.zeros([N], dtype=tf.int64))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

test = []
for i, j in zip(te_data, te_feature): #元素个数与最短的列表一致
    distances = tf.reduce_sum(tf.square(i- tr_data), reduction_indices=1) #计算te_data中某个样本与tr_data中所有样本的距离值
    neighor = tf.argmin(distances,axis=0) # te_data中该样本与tr_data中距离最小的样本所在的索引
    # print("the shape of distance = ", sess.run(tf.shape(distances)))
    test.append(tr_features[sess.run(neighor)])

print(test)
fig, ax = plt.subplots()
ax.scatter(te_data.transpose()[0], te_data.transpose()[1], marker = 'o', s = 100, c = test, cmap=plt.cm.coolwarm )
plt.show()
