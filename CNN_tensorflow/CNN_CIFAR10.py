# -*- coding: utf-8 -*-
# Time    : 2018/9/11/0011 12:00
# Author  :liangye
# FileName: CNN_CIFAR10.py

import tensorflow as tf
from  tensorflow.contrib import learn
import glob
import numpy as np
from sklearn import metrics
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# parameters
learning_rate = 0.001
training_iters = 7000
batch_size = 100
display_step = 10

# network parameters
n_classes = 10 # 10 digit classes
dropout = 0.80 # Dropout probability
test_input = 3000

sess = tf.Session()
datadir='data/cifar-10-batches-bin/' # the book use 10000 images to construct  datasets
G = glob.glob (datadir + '*.bin')
A = np.fromfile(G[0],dtype=np.uint8).reshape([10000,3073]) # note that the first col is label
# print(A.shape) # (10000, 3073)
# labels = A[:,0]
labels = tf.convert_to_tensor(A[:,0],dtype=tf.uint8)
images = tf.convert_to_tensor(A[:,1:].reshape([10000,3,32,32]).transpose(0,2,3,1),dtype=tf.uint8) # 了解数据的存储形式, 并把通道的维度放到最后
labels_OneHot = tf.one_hot(labels,10,on_value=1,off_value=0)
# print('labels_OneHot = \n', sess.run(labels_OneHot))
# print('shape of images = \n', sess.run(tf.shape(images)))

# split the train and test
train_images = sess.run(tf.slice(images,[0,0,0,0],[7000,32,32,3]))
train_labels = sess.run(tf.slice(labels_OneHot,[0,0],[7000,10]))

test_images = sess.run(tf.slice(images,[7000,0,0,0],[3000,32,32,3]))
test_labels = sess.run(tf.slice(labels_OneHot,[7000,0],[3000,10]))

# print('shape of train_labels = \n', sess.run(tf.shape(test_images)))
# print('shape of train_images = \n', sess.run(tf.shape(test_labels)))

X = tf.placeholder(tf.float32,[None,32,32,3]) # After X passed to conv2d, X will be reshape
Y = tf.placeholder(tf.float32,[None,n_classes])
keep_prob = tf.placeholder(tf.float32)

# weights and biases 的维度确定, 受下面padding,stride的影响
# 该模型中卷积操作padding=same, stride=1, pooling操作中窗口大小ksize=2(即窗口大小2*2),stride=2
weights = {
    # 权重的维度: [卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
    'wc1' : tf.Variable(tf.random_normal([5,5,3,32])), # note that this kernel size is 5*5, output 32
    # 5x5 convolutional units, 32 inputs, 64 outputs
    'wc2' : tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd' : tf.Variable(tf.random_normal([8*8*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out' : tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# define some function for simplicity
def conv2d(x,w,b,strides=1): # create a block which achive convolution, bias, activation--relue.
    x = tf.nn.conv2d(x,w,strides=[1,strides,strides,1],padding='SAME')
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)

def max_pooling(x,k):
    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

# construct the network structure
def conv_net(x,w,b,dropout_value):
    '''
    define a model which has 3 conv layers and 2 fully connected layers
    :return:
    '''
    x = tf.reshape(x,[-1,32,32,3]) # note that the original image data is 32*32

    # conv layer
    conv1 = conv2d(x, w['wc1'], b['bc1'])  # use the conv2d function, use default param strides=1
    conv1 = max_pooling(conv1, k=2)

    # conv layer
    conv2 = conv2d(conv1, w['wc2'], b['bc2'])
    conv2 = max_pooling(conv2, k=2)

    # fully conneted layer
    fc1 = tf.reshape(conv2, [-1, w['wd'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, w['wd']), b['bd'])
    fc1 = tf.nn.relu(fc1)
    # apply dropout
    fc1 = tf.nn.dropout(fc1, keep_prob=dropout_value)

    # output layer
    out_rs = tf.add(tf.matmul(fc1, w['out']), b['out'])
    return out_rs

# use model
pred_y = conv_net(X,weights,biases,keep_prob)
# define the cost and optimal
# some codes use softmax in model, define the cost using cross entropy(these are 2 stages)
# but here softmax_cross_entropy_with_logits can achive the same resutls
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_y,labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# evaluate the model
correct_pred = tf.equal(tf.argmax(pred_y,1),tf.argmax(Y,1))
acc = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

sess.run(tf.global_variables_initializer())
step = 1
batch_ind = 0

while step*batch_size < training_iters: # step is up to 69, and the last batch is 6900~7000
    if batch_ind == 0:
        batch_ind = step * batch_size - 1
    else:
        batch_ind = step * batch_size

    batch_x,batch_y = sess.run(tf.slice(train_images,[batch_ind,0,0,0],[batch_size,32,32,3])), sess.run(tf.slice(train_labels,[batch_ind,0],[batch_size,10]))
    sess.run(optimizer,feed_dict={X:batch_x,Y:batch_y,keep_prob: dropout})
    if step*batch_size%100==0:
        loss,Acc = sess.run([cost,acc],feed_dict={X:batch_x,Y:batch_y,keep_prob:1.}) # here just compute the train cost and acc, which no need to use dropout
        print('Iteration: ', step*batch_size, ', Minibatch Loss= '+ "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.4f}".format(Acc), ", batch_ind=", batch_ind)
    step += 1
    batch_ind+=1

print(" Finish training and optimizing model!--------------------------------------------")
print("Test Acc is = ", sess.run(acc, feed_dict={X: test_images, Y: test_labels, keep_prob: 1.})) # test model no use to use dropout






