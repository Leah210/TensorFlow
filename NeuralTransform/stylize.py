# -*- coding: utf-8 -*-
# Time    : 2018/09/26 16:03
# Author  :liangye
# FileName: stylize.py

import vgg
import tensorflow as tf
import numpy as np
from sys import stderr
from PIL import Image
from functools import reduce
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 定义content和style特征的层
CONTENT_LAYERS = ('relu4_2', 'relu5_2')
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

def stylize(network, initial, initial_noiseblend, content, styles, preserve_colors,
            iterations,content_weight, content_weight_blend, style_weight,
            style_layer_weight_exp, style_blend_weights, tv_weight,learning_rate,
            beta1, beta2, epsilon, pooling, print_iterations=None, checkpoint_iterations=None):
    '''
    该函数包括利用初始参数提取content和style特征,并进行根据style特征对图像进行重构
    :param network:
    :param initial:
    :param initial_noiseblend:
    :param content: 经过读取后的content图片
    :param styles:  为风格图片集, 可以为多张图片
    :param preserve_colors:
    :param iterations: 迭代次数
    :param content_weight: 内容损失的系数
    :param content_weight_blend: 两个content重构层的占比, 默认为1, 只使用更精细的重构层relu4_2；更抽象的重构层relu5_2占比为1-content_weight_blend.
    :param style_weight:
    :param style_layer_weight_exp:
    :param style_blend_weights: 风格图片集之间的权重
    :param tv_weight:
    :param learning_rate:
    :param beta1:
    :param beta2:
    :param epsilon:
    :param pooling: 池化的方式,max or avg, 代码中已经指定好, 默认参数POOLING='max'
    :param print_iterations:
    :param checkpoint_iterations:
    :return:
    '''
    # input shape: [batch, height, width, channels], only one image, so batch=1.
    shape = (1,) + content.shape # 内容的shape增加一个维度
    style_shape = [(1,) + style.shape for style in styles] # 风格每个通道的shape增加一个维度

    content_features = {} # 创建content feature的字典
    style_features = [{} for _ in styles] # 由于有多张style图片, 每个图片创建字典, 即得到style feature的字典list

    vgg_weights, vgg_mean_pixel = vgg.load_net(network) # mat给定的路径, 使用时option.network, 而network的默认参数为VGG_PATH

    layer_weight = 1.0
    style_layer_weights = {} # 不同神经网络层的权重
    for style_layer in STYLE_LAYERS:
        style_layer_weights[style_layer] = layer_weight # 指定style层的初始权重均为1
        style_weight *= style_layer_weight_exp # style_weight 是指数级增长的

    # 标准化style_layer_weights
    layer_weight_sum = 0.0
    for style_layer in STYLE_LAYERS:
        layer_weight_sum += style_layer_weights[style_layer] # 所有style layer的权重之和
    for style_layer in STYLE_LAYERS:
        style_layer_weights[style_layer] /= layer_weight_sum # 每层权重的占比

    sess = tf.Session()
    image = tf.placeholder('float', shape=shape)
    net = vgg.net_preload(vgg_weights,image,pooling) # 池化的方式,max or avg, 代码中已经指定好, 默认参数POOLING='max',利用初始参数和图片(待定)构建网络
    # 利用前馈计算content features
    content_pre = np.array([vgg.preprocess(content, vgg_mean_pixel)]) # 归一化的content
    for layer in CONTENT_LAYERS:
        # content_features[layer] = net[layer].eval(feed_dict={image:content_pre})
        content_features[layer] = sess.run(net[layer], feed_dict={image: content_pre}) # 与上面代码一样
    # 也可以这样子实现
    # content_features[CONTENT_LAYERS] = sess.run(net[CONTENT_LAYERS], feed_dict={image: content_pre})

    # 利用前馈计算style features,因为有多个图片,每个通道都提取features
    for i in range(len(styles)): # 循环多张图片
        # For relu1_1 layer, features: (1, 112, 112, 64)
        image = tf.placeholder('float', shape=style_shape[i]) # 例如style_shape[0]为(1,224,224,3)
        net = vgg.net_preload(vgg_weights,image,pooling)
        # 利用前馈计算style features
        style_pre = np.array([vgg.preprocess(styles[i], vgg_mean_pixel)])
        for layer in STYLE_LAYERS: # ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
            features = net[layer].eval(feed_dict = {image: style_pre},session = sess) # 计算当前图片的style features
            features = np.reshape(features,(-1,features.shape[3])) # features.shape[3] is the number of filters
            # 例如 当前卷积操作之后得到112*112*64的结果, 那么经过reshape之后,feature的维度为(12544, 64)
            gram = np.dot(features.T, features)/features.size  # (64, 64) Gram matrix (features.size, 元素个数)
            style_features[i][layer] = gram

    # initial_noiseblend, ratio of blending initial image with normalized noise
    # (if no initial image specified, content image is used)
    initial_noiseblend_coef = 1.0 - initial_noiseblend

    # 图像重构部分
    noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
    if initial is None:
        initial = tf.random_normal(shape) * 0.256 # 如果没有指定initial input image, 则生成随机图片(暂时不理解这里为什么*0.256)
    else:
        initial = np.array([vgg.preprocess(initial,vgg_mean_pixel)]).astype('float32')
        initial = initial * initial_noiseblend_coef + (tf.random_normal(shape) * 0.256)* initial_noiseblend # initial input image blend with noise
    image = tf.Variable(initial)
    net = vgg.net_preload(vgg_weights, image, pooling) # 此处的net保存的是initial image的模型结果


    # content loss
    content_layers_weight = {}
    content_layers_weight['relu4_2'] = content_weight_blend #content_weight_blend, 两个content重构层的占比, 默认为1
    content_layers_weight['relu5_2'] = 1.0 - content_weight_blend

    content_loss = 0.0 # 总的content损失
    content_losses = [] # 每层layer的损失
    for layer in CONTENT_LAYERS:
        content_losses.append(content_layers_weight[layer] * content_weight
                              + ( 2 * tf.nn.l2_loss( net[layer] - content_features[layer] ) / content_features[layer].size)
                              ) # 由于tf.nn.l2_loss是1/2(A-B)^2,因此这里*2, 再进行归一化
    content_loss += reduce(tf.add, content_losses)

    # style loss
    style_loss = 0.0
    for i in range(len(styles)):
        style_losses = []
        for style_layer in STYLE_LAYERS:
            layer = net[style_layer] # 提取指定层的结果
            _, height, width, number = map(lambda i: i.value, layer.get_shape()) # layer.get_shape()得到的结果为当幅图(tensor)在当层的结果尺寸, 如112*112*64
            size = height * width * number
            feats = tf.reshape(layer,(-1,number))
            gram = tf.matmul(tf.transpose(feats), feats) / size # 由于此处layer是tensor, 这里需要用tensor的方法来处理, 此处gram是initial image的gram
            style_gram = style_features[i][style_layer]
            # style_losses.append(style_layer_weights[style_layer]  * 2 * tf.nn.l2_loss(gram - style_gram)/style_features[i][style_layer].size)
            style_losses.append(style_layer_weights[style_layer] * 2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size)
        style_loss += style_weight*style_blend_weights[i]*reduce(tf.add, style_losses)


    # total variation denoising
    tv_x_size = tensor_size(image[:, 1:, :, :]) # image的shape为(1,height,width,number), 注意image是tensor
    tv_y_size = tensor_size(image[:, :, 1:, :])

    # 这里计算的是像素点之间的差异, 先去理解总变分的计算公式
    tv_loss = tv_weight * 2 * (
        (tf.nn.l2_loss(image[:, 1:, :, :] - image[:, :shape[1]-1:, :, :])/tv_y_size)
        + (tf.nn.l2_loss(image[:, :, 1:, :] - image[:, :, :shape[2] - 1:, :])/tv_x_size)
    )

    # total loss
    loss = content_loss + style_loss + tv_loss
    train_step = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = beta1, beta2 = beta2, epsilon = epsilon).minimize(loss = loss)

    # 调试打log
    def print_progress():
        stderr.write('  content loss: %g\n' % content_loss.eval(session = sess))
        stderr.write('    style loss: %g\n' % style_loss.eval(session = sess))
        stderr.write('       tv loss: %g\n' % tv_loss.eval(session = sess))
        stderr.write('    total loss: %g\n' % loss.eval(session = sess))

    # optimization
    best_loss = float('inf')
    best = None # 记录下重构的图像
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    stderr.write("Optimization started now....")

    for i in range(iterations):
        stderr.write("Iteration %4d/%4d\n" %(i+1, iterations))
        sess.run(train_step) # 或者 train_step.run()
        last_step = (i == iterations-1)

        if last_step or (print_iterations and i%iterations==0):
            print_progress()
        if (checkpoint_iterations and i%checkpoint_iterations==0) or last_step:
            this_loss = loss.eval(session = sess)  # 或者直接sess.run(loss), 由于在net_preload()中已经feed参数了, 因此这里不需要再feed
            if this_loss < best_loss:
                best_loss = this_loss
                best = image.eval(session = sess)
            # 重构输出图片
            img_out = vgg.unprocess(best.reshape(shape[1:]), vgg_mean_pixel)
            # if preserve_colors and preserve_colors==True:
            #     original_image = np.clip(content,0,255) # 把content中的像素规范到0-255之间, 即小与0的取0,大于255的取255
            #     style_image = np.clip(img_out,0,255)  # 把img_out中的像素规范到0-255之间, 即小与0的取0,大于255的取255

            yield (
                (None if last_step else i), img_out
            )

def tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1) # 实际上类似于size, 例如得到tensor的shape为(112,112,64), 最终return的结果为112*112*64=802816

# note:-----------------------------------------------------------------------------------------------------------------
# tensor.eval()
        # 1.eval取得的是在tesnor.eval()之前对tensor所做的所有操作的最后结果
        # 2.eval()不会累加计算, {tensor.eval()} 与{tensor.eval();tensor.eval()}的返回的numpy 类型的ndarray是相同的
        # 3.eval() 与 sess.run()的结果是完全一致的
    
# 从数学上定义什么是风格, 是Neural Style比较有意思的地方. 每个卷积核filter可以看做是图形的一种特征抽取.
    # 风格在这篇paper中被简化为任意两种特征的相关性. 相关性的描述使用余弦相似性, 而余弦相似性又正比于两种特征的点积.
    # 于是风格的数学定义被表示为神经网络层里filter i和filter j的点积, 用G_{ij}^l表示

# 格拉姆矩阵可以看做feature之间的偏心协方差矩阵(即没有减去均值的协方差矩阵),在feature map中, 每个数字都来自于一个特定滤波器在特定位置的卷积,
        # 因此每个数字代表一个特征的强度, 而Gram计算的实际上是两两特征之间的相关性, 哪两个特征是同时出现的, 哪两个是此消彼长的等等,
        # 同时, Gram的对角线元素, 还体现了每个特征在图像中出现的量, 因此, Gram有助于把握整个图像的大体风格.
        # 有了表示风格的Gram Matrix, 要度量两个图像风格的差异, 只需比较他们Gram Matrix的差异即可.
        # 总之,  格拉姆矩阵用于度量各个维度自己的特性以及各个维度之间的关系. 内积之后得到的多尺度矩阵中, 对角线元素提供了不同特征图各自的信息,
        # 其余元素提供了不同特征图之间的相关信息. 这样一个矩阵, 既能体现出有哪些特征, 又能体现出不同特征间的紧密程度.
# 例如:
# x1=[3,3]
# x2 = [4, 3]
# x3 = [1, 1]
    # G1 =[
    # 18 21 6
    # 21 25 7
    # 6 7 2
    # ] 对角线可以看出这3个特征重要程度, 其他元素看出元素之间的相关性

# x11 = [1,2,3]
# x22 = [1,2,3]
# x33 = [1,2,3]
    # G1 =[
    # 14 14 14
    # 14 14 14
    # 14 14 14
    # ] 对角线就说明这3个特征重要程度是一样的, 以及它们之间的相关性是一样的

# 用传给reduce中的函数 func()(必须是一个二元操作函数)先对集合中的第1,2个数据进行操作,得到的结果再与第三个数据用func()函数运算,最后得到一个结果

# 由于模型优化的是一张随机的白噪声图片, 因此该白噪声图片会不断拟合content和style图片,
    # 模型优化是是白噪声图片中的weight和bias, 而不是初始模型中的weight和bias,
    # 总损失函数中可以加入总变分噪声(total variation denoising), 也可以不加入, 目的在于平滑信号, Add cost to penalize neighboring pixel is very different.
    # 总体来讲，由于噪音信号（图像）相邻信号间不平滑，随机变化较大，故total variation比较大，而平滑信号则TV项比较小。通过minimize total variation，可以去除噪音，平滑信号。
    # 因此加入total variation denoising的话, 在理论和实验结果上应该效果会更好

# tensor.get_shape() # 获取已知张量的维度, 输出为一个元组
# tf.shape(x) # 其中x可以是tensor, list, array

# yeil and return:
# 功能都是返回程序执行结果, yield返回执行结果并不中断程序执行，return在返回执行结果的同时中断程序执行