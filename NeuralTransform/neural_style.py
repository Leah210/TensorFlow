# -*- coding: utf-8 -*-
# Time    : 2018/09/28 15:46
# Author  :liangye
# FileName: neural_style.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import stylize
import math
import scipy.misc
from argparse import ArgumentParser
from PIL import Image

# 定义默认参数
CONTENT_WEIGHT = 5e0
CONTENT_WEIGHT_BLEND = 1
STYLE_WEIGHT = 5e2
TV_WEIGHT = 1e2
STYLE_LAYER_WEIGHT_EXP = 1
LEARNING_RATE = 1e1
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
STYLE_SCALE = 1.0
ITERATIONS = 1000
VGG_PATH = 'D:/BaiduNetdiskDownload/imagenet-vgg-verydeep-19.mat'
POOLING = 'max'

# 定义参数解析函数
def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content',
                        dest = 'content', help = 'content image',
                        metavar = 'CONTENT', required = True)
    parser.add_argument('--styles',
                        dest = 'styles',
                        nargs = '+', help = 'one or more style images',
                        metavar = 'STYLE', required = True)
    parser.add_argument('--output',
                        dest = 'output', help = 'output path',
                        metavar = 'OUTPUT', required = True)
    parser.add_argument('--iterations', type = int,
                        dest = 'iterations', help = 'iterations (default %(default)s)',
                        metavar = 'ITERATIONS', default = ITERATIONS)
    parser.add_argument('--print-iterations', type = int,
                        dest = 'print_iterations', help = 'statistics printing frequency',
                        metavar = 'PRINT_ITERATIONS')
    parser.add_argument('--checkpoint-output',
                        dest = 'checkpoint_output', help = 'checkpoint output format, e.g. output%%s.jpg',
                        metavar = 'OUTPUT')
    parser.add_argument('--checkpoint-iterations', type = int,
                        dest = 'checkpoint_iterations', help = 'checkpoint frequency',
                        metavar = 'CHECKPOINT_ITERATIONS')
    parser.add_argument('--width', type = int,
                        dest = 'width', help = 'output width',
                        metavar = 'WIDTH')
    parser.add_argument('--style-scales', type = float,
                        dest = 'style_scales',
                        nargs = '+', help = 'one or more style scales',
                        metavar = 'STYLE_SCALE')
    parser.add_argument('--network',
                        dest = 'network', help = 'path to network parameters (default %(default)s)',
                        metavar = 'VGG_PATH', default = VGG_PATH)
    parser.add_argument('--content-weight-blend', type = float,
                        dest = 'content_weight_blend',
                        help = 'content weight blend, conv4_2 * blend + conv5_2 * (1-blend) (default %(default)s)',
                        metavar = 'CONTENT_WEIGHT_BLEND', default = CONTENT_WEIGHT_BLEND)
    parser.add_argument('--content-weight', type = float,
                        dest = 'content_weight', help = 'content weight (default %(default)s)',
                        metavar = 'CONTENT_WEIGHT', default = CONTENT_WEIGHT)
    parser.add_argument('--style-weight', type = float,
                        dest = 'style_weight', help = 'style weight (default %(default)s)',
                        metavar = 'STYLE_WEIGHT', default = STYLE_WEIGHT)
    parser.add_argument('--style-layer-weight-exp', type = float,
                        dest = 'style_layer_weight_exp',
                        help = 'style layer weight exponentional increase - weight(layer<n+1>) = weight_exp*weight(layer<n>) (default %(default)s)',
                        metavar = 'STYLE_LAYER_WEIGHT_EXP', default = STYLE_LAYER_WEIGHT_EXP)
    parser.add_argument('--style-blend-weights', type = float,
                        dest = 'style_blend_weights', help = 'style blending weights',
                        nargs = '+', metavar = 'STYLE_BLEND_WEIGHT')
    parser.add_argument('--tv-weight', type = float,
                        dest = 'tv_weight', help = 'total variation regularization weight (default %(default)s)',
                        metavar = 'TV_WEIGHT', default = TV_WEIGHT)
    parser.add_argument('--learning-rate', type = float,
                        dest = 'learning_rate', help = 'learning rate (default %(default)s)',
                        metavar = 'LEARNING_RATE', default = LEARNING_RATE)
    parser.add_argument('--beta1', type = float,
                        dest = 'beta1', help = 'Adam: beta1 parameter (default %(default)s)',
                        metavar = 'BETA1', default = BETA1)
    parser.add_argument('--beta2', type = float,
                        dest = 'beta2', help = 'Adam: beta2 parameter (default %(default)s)',
                        metavar = 'BETA2', default = BETA2)
    parser.add_argument('--eps', type = float,
                        dest = 'epsilon', help = 'Adam: epsilon parameter (default %(default)s)',
                        metavar = 'EPSILON', default = EPSILON)
    parser.add_argument('--initial',
                        dest = 'initial', help = 'initial image',
                        metavar = 'INITIAL')
    parser.add_argument('--initial-noiseblend', type = float,
                        dest = 'initial_noiseblend',
                        help = 'ratio of blending initial image with normalized noise (if no initial image specified, content image is used) (default %(default)s)',
                        metavar = 'INITIAL_NOISEBLEND')
    parser.add_argument('--preserve-colors', action = 'store_true',
                        dest = 'preserve_colors',
                        help = 'style-only transfer (preserving colors) - if color transfer is not needed')
    parser.add_argument('--pooling',
                        dest = 'pooling', help = 'pooling layer configuration: max or avg (default %(default)s)',
                        metavar = 'POOLING', default = POOLING)
    parser.add_argument('--overwrite', action = 'store_true',
                        dest = 'overwrite', help = 'write file even if there is already a file with that name')
    return parser

def main():
    parser = build_parser()
    options = parser.parse_args()

    content_image = imread(options.content) # 命令行中参数传进来
    style_images = [imread(style_img) for style_img in options.styles] # 命令行中传进的参数

    width = options.width
    if width is not None: # 假设有width参数传进来, 那么content的大小要重新设置, 即按照比例进行缩放
        # 如height/weight = new_h/new_w, 其中new_w是给定的
        # 则new_h = (height/weight)/new_w
        new_shape = (int(math.floor(float(content_image.shape[0]) / content_image.shape[1] * width)), width)
        content_image = scipy.misc.imresize(content_image,new_shape)
    target_shape = content_image.shape # 生成图片的大小
    for i in range(len(style_images)):
        style_scale = STYLE_SCALE
        if options.style_scales is not None: # 若style的尺度有指定, 则需要对styles图片集进行缩放
            style_scale = options.style_scales[i] # 每幅图都制定了尺度
        # imresize 传进的size参数
        # size: int, float or tuple
        # *int - Percentage of current size.
        # *float - Fraction of current size.
        # *tuple - Size of the output image(height, width).
        # 那么这里使用的是float, 尺寸: width/height
        style_images[i] = scipy.misc.imresize(style_images[i], style_scale * target_shape[1] / style_images[i].shape[1]) # 每张图按照指定尺度读入

    style_blend_weights = options.style_blend_weights
    if style_blend_weights is None:
        # 设定默认的相等的weights
        style_blend_weights = [1.0/len(style_images) for _ in style_images]
    else:
        total_blend_weight = sum(style_blend_weights)
        style_blend_weights = [weight/total_blend_weight for weight in style_blend_weights] # 对style_blend_weights进行标准化

    initial = options.initial
    if initial is not None: # 表明生成图像是指定的
        initial = scipy.misc.imresize(imread(initial), content_image.shape[:2]) # 生成图像的大小需要进行变换
        if options.initial_noiseblend is None:
            options.initial_noiseblend = 0.0
    else: # 如果没有提供初始图,也没有提供初始的noiseblend,则生成随机噪声图片
        if options.initial_noiseblend is None:
            options.initial_noiseblend = 1.0
        if options.initial_noiseblend < 1.0:
            initial = content_image
    if options.checkpoint_output and "%s" not in options.checkpoint_output:
        parser.error("To save intermediate images, the checkpoint output parameter must contain '%s' (e.g. `foo%s.jpg`)")

    # 为了保证图像可以写入,设置写入虚拟图像
    if os.path.isfile(options.output) and not options.overwrite:
        raise IOError("%s already exists, will not replace it without the '--overwrite' flag" % options.output)
    try:
        imsave(options.output, np.zeros((500,500,3)))
    except:
        raise IOError('%s is not writable or does not have a valid file extension for an image file' % options.output)

    for iteration, image in stylize.stylize(network = options.network,initial = initial,initial_noiseblend = options.initial_noiseblend,
                                            content = content_image,styles = style_images,preserve_colors = options.preserve_colors,
                                            iterations = options.iterations,content_weight = options.content_weight,
                                            content_weight_blend = options.content_weight_blend,style_weight = options.style_weight,
                                            style_layer_weight_exp = options.style_layer_weight_exp,style_blend_weights = style_blend_weights,
                                            tv_weight = options.tv_weight,learning_rate = options.learning_rate,beta1 = options.beta1,
                                            beta2 = options.beta2,epsilon = options.epsilon, pooling = options.pooling,
                                            print_iterations = options.print_iterations, checkpoint_iterations = options.checkpoint_iterations):
        # stylize 每一次迭代返回一个iteration and image
        output_file = None
        combined_rgb = image
        if iteration is not None:
            if options.checkpoint_output:
                output_file = options.checkpoint_output % iteration
        else:
            output_file = options.output
        if output_file:
            imsave(output_file,combined_rgb)

def imread(path):
    '''
    自定义imread函数,覆写scipy.misc中的imread
    :param path:
    :return:
    '''
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img, img, img))
    elif img.shape[2] == 4:
        # PNG with alpha channel
        img = img[:, :, :3]
    return img

def imsave(path,img):
    '''
    自定义imsave函数,覆写scipy.misc中的imsave
    :param path:
    :param img:
    :return:
    '''
    img = np.clip(img, 0, 255).astype(np.uint8) #  数据小与0则变成0, 数据大于255则变成255
    Image.fromarray(img).save(path,quality = 95)

if __name__ == '__main__':
    main()