# -*- coding: utf-8 -*-
# Time    : 2018/9/10/0010 11:18
# Author  :liangye
# FileName: CNN_MNIST.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy
from tensorflow.examples.tutorials.mnist import input_data

sess = tf.Session()
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# parameters
learning_rate = 0.001
training_iters = 2000
batch_size = 128
display_step = 10

# network parameters
n_input = 784 #image size (28*28)
n_classes = 10 # 10 digit classes
dropout = 0.80 # Dropout probability

X = tf.placeholder(tf.float32,[None,n_input]) # After X passed to conv2d, X will be reshape
Y = tf.placeholder(tf.float32,[None,n_classes])
keep_prob = tf.placeholder(tf.float32)

weights = {
    # 5x5 convolutional units, 1 input, 32 outputs
    # [卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
    # wc1卷积核个数相当于下一层的通道个数
    'wc1' : tf.Variable(tf.random_normal([5,5,1,32])),
    # 5x5 convolutional units, 32 inputs, 64 outputs
    'wc2' : tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd' : tf.Variable(tf.random_normal([7*7*64, 1024])),
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
    # note that w is a filter with [filter_height, filter_width, in_channels, out_channels]
    x = tf.nn.conv2d(x,w,strides=[1,strides,strides,1],padding='SAME')
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)

# 池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
# 那么这里pooling的窗口大小为2*2,stride=2
def max_pooling(x,k):
    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

def conv_net(x,w,b,dropout_value):
    '''
    2 conv layers, 1 fully connected layer
    '''
    # reshape the input
    x = tf.reshape(x, [-1,28,28,1])
    # conv layer
    conv1 = conv2d(x,w['wc1'],b['bc1']) # use the conv2d function
    conv1 = max_pooling(conv1,k=2) #
    print("the shape of conv1 is =", tf.shape(conv1))
    # conv layer
    conv2 = conv2d(conv1,w['wc2'],b['bc2'])
    conv2 = max_pooling(conv2,k=2)
    print("the shape of conv2 is =", tf.shape(conv2))
    # fully conneted layer
    fc1 = tf.reshape(conv2,[-1,w['wd'].get_shape().as_list()[0]])
    print("the shape of fc1 in first step is =", tf.shape(fc1))
    fc1 = tf.add(tf.matmul(fc1,w['wd']),b['bd'])
    fc1 = tf.nn.relu(fc1)
    # apply dropout
    fc1 = tf.nn.dropout(fc1, keep_prob=dropout_value)
    #output layer
    out_rs = tf.add(tf.matmul(fc1,w['out']),b['out'])
    return out_rs

# use model
pred_y = conv_net(X,weights,biases,keep_prob)
# define the cost and optimal
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_y,labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# evaluate the model
correct_pred = tf.equal(tf.argmax(pred_y,1),tf.argmax(Y,1)) # due to one_hot encode the label
acc = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

sess.run(tf.global_variables_initializer())
step = 1

# step*batch_size < training_iters
while step*batch_size < training_iters: # control the train iteration
    batch_x,batch_y = mnist.train.next_batch(batch_size)
    test = batch_x[0]
    sess.run(optimizer,feed_dict={X:batch_x,Y:batch_y, keep_prob: dropout}) # note the keep_prob here.
    if step % display_step == 0:
        loss, Acc = sess.run([cost,acc], feed_dict={X:batch_x,Y:batch_y,keep_prob:1.}) # note that cost needs pred_y-X,Y,keep_prob. Here keep_proc=1 account for didn't use dropout
        print(
        "Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
        "{:.6f}".format(loss) + ", Training Accuracy= " + \
        "{:.5f}".format(Acc))
    step += 1
print("--------------------------------------------\n Finish training and optimizing model!")
print('the shape of test_y is = ', sess.run(tf.shape(mnist.test.images[0:256])))
print("Test Acc is = ", sess.run(acc, feed_dict={X: mnist.test.images[:256], Y: mnist.test.labels[:256], keep_prob: 1.})) # test model no use to use dropout

# convolution detail----------------------------------------------------------------------------------------------------
# conv1: 28*28*1 -- 5*5*1 (32个卷积核), stride=1,padding=same, 向上取整 → 28*28*32
# max_pool(2*2,stride=2,padding=same) → 14*14*32
# conv2: 14*14*32 -- 5*5*32 (64个卷积核), stride=1, padding=same, 向下取整 → 14*14*64
# max_pool(2*2,stride=2,padding=same) → 7*7*64
# fc1 : 将7*7*64的conv2展开后, 传递到 1024个单元的全连接层 → 那么wd的维度为: 3136 * 1024
# out : 输出n_class个单元(由于one_hot编码), 则out的维度为1024*10

# padding detail--------------------------------------------------------------------------------------------------------
# 1、如果padding = ‘VALID’
# new_height = new_width = (W – F + 1) / S （结果向上取整）
# 2、如果padding = ‘SAME’
# new_height = new_width = W / S （结果向上取整）

# TensorFlow detail-----------------------------------------------------------------------------------------------------
# note the output acc is the same name as tensor acc, it would make a mistake: Can not convert a float32 into a Tensor or Operation.
# so the tensor turn to float32
# When I need to use tensor acc, it will show i can not convert a float32 into a Tensor or Operation.
# the original code is below:
# loss, acc = sess.run([cost,acc], feed_dict={X:batch_x,Y:batch_y,keep_prob:1.}) # note that cost needs pred_y-X,Y,keep_prob. Here keep_proc=1 account for didn't use dropout
# print( "Iter " + str(step * batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
# So I just modify the variable name output acc as Acc, like this:
# loss, Acc = sess.run([cost,acc], feed_dict={X:batch_x,Y:batch_y,keep_prob:1.}) # note that cost needs pred_y-X,Y,keep_prob. Here keep_proc=1 account for didn't use dropout
# print( "Iter " + str(step * batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(Acc))





