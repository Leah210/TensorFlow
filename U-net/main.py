# -*- coding: utf-8 -*-
# Time    : 2018/11/06 14:26
# Author  :liangye
# FileName: main.py
import tensorflow as tf
import numpy as np
from unet import *
import scipy.misc as sm
import os
from generate_img import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TRAIN_IMAGE_PATH = 'D:/PycharmProjects/U-net/train/image/'
TRAIN_LABEL_PATH = 'D:/PycharmProjects/U-net/train/label/'
TRAIN_AUG_PATH = 'D:/PycharmProjects/U-net/train/augmentation/'
TEST_PATH = 'D:/PycharmProjects/U-net/test/'


if __name__ == "__main__":
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    weights = sess.run(weights)
    biases = sess.run(biases)
    decov_weights = sess.run(decov_weights)

    BETA1 = 0.9
    BETA2 = 0.999
    EPOCH = 1  # use whole dataset epoch times
    BATCH_SIZE = 32
    EPSILON = 1e-08
    ITERATIONS = 100  # if batch_size=128 and epoch=1, then the number of sample is 12800

    # parameters
    learning_rate = 0.1
    training_iters = 25
    batch_size = 2
    # display_step = 10

    # network parameters
    n_classes = 2  # 10 digit classes
    dropout = 0.70  # Dropout probability
    test_input = 3000
    image_size_w = 64#512
    image_size_h = 64#512

    train_image, train_label = train_set_up(TRAIN_IMAGE_PATH, TRAIN_LABEL_PATH, TRAIN_AUG_PATH,(image_size_w,
                                                                                                image_size_h))
    test_image, test_label = test_set_up(TEST_PATH,(image_size_w, image_size_h))

    step = 20
    batch_ind = 0
    # X = tf.placeholder(tf.float32, shape = [batch_size,image_size_w,image_size_h,1])
    # Y = tf.placeholder(tf.float32, shape = [batch_size,image_size_w,image_size_h,1])

    while step < training_iters:
        print('In the {}'.format(step),'trainning iteration~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        batch_ind = step * batch_size - batch_size

        batch_x = tf.slice(train_image, [batch_ind, 0, 0, 0], [batch_size, image_size_w, image_size_h, 1])
        batch_y = tf.slice(train_label, [batch_ind, 0, 0, 0], [batch_size, image_size_w, image_size_h, 1])

        feature_map = unet(batch_x, weights = weights, biases = biases, keep_prob = dropout)
        loss = compute_loss(feature_map = feature_map, label = batch_y)

        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss = loss)  # Since AdamOptimizer has it's own variables,  should define the initilizer init after opt, not before.
        sess.run(tf.global_variables_initializer())

        cost = sess.run(loss)
        # print('In this iteration, the loss is = ', cost)
        opt= sess.run([optimizer])
        # print('the optimaizer had been trained!!')

        # if step % 2 == 0:
        _, acc_tensor = accuracy(feature_map, batch_y)
        acc = sess.run(acc_tensor)
        print('Minibatch Loss= ' + "{:.4f}".format(cost) + ", Training Accuracy= " + "{:.4f}".format(acc))

        step += 1
    print("Finish training and optimizing model!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # test the model
    print('\n\n')
    print('Enter the test step!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    pred_sum = []
    pred_acc = []
    for i in range(5): #sess.run(tf.shape(test_image)[0])
        test_img = tf.slice(test_image,[i,0,0,0],[1, image_size_w, image_size_h, 1])
        test_l = tf.slice(test_label,[i,0,0,0],[1, image_size_w, image_size_h, 1])
        test_predict_map = unet(test_image, weights = weights, biases = biases, keep_prob = dropout)
        test_pred, test_acc_invidual = accuracy(test_predict_map, test_l)
        test_p = sess.run(test_pred)
        test_a = sess.run(test_acc_invidual)
        pred_sum.append(test_p)
        pred_acc.append(test_a)
        print('In this iteration of test, the accruacy is {:.4f}'.format(test_a))
    test_pre_acc = np.mean(pred_sum)
    print("Test Predict Acc is = ", test_pre_acc*100, "%")  # test model no use to use dropout
    test_acc = np.mean(pred_acc)
    print('Test Mean of Iinvidual Acc is = ', test_acc * 100, "%")
    # save the model
    all_parameters_saver = tf.train.Saver()
    all_parameters_saver.save(sess = sess, save_path = './checkpoint/')
    print('save the trained model!-----------------------------------------------------------')



#