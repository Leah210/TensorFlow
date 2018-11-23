# -*- coding: utf-8 -*-
# Time    : 2018/11/01 10:54
# Author  :liangye
# FileName: generate_img.py

# generate images for training the model using keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import os
import scipy.misc as sc
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# define train data augmentation
data_gen_args = dict(featurewise_center = True,
                     featurewise_std_normalization = True,
                     rotation_range = 90.,
                     width_shift_range = 0.1,
                     height_shift_range = 0.1,
                     zoom_range = 0.2)
def train_data_augmentation(batch_size, train_path, image_folder, mask_folder, image_color_mode = "grayscale",
                          mask_color_mode = "grayscale", image_save_prefix = "image", mask_save_prefix = "mask",
                          num_class = 2, save_to_dir = None, target_size = (512, 512),
                          image_num = 10, seed = 1):
    data_gen = ImageDataGenerator(
            rotation_range = 40,
            width_shift_range = 0.2,
            height_shift_range = 0.2,
            rescale = 1. / 255,
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True,
            fill_mode = 'nearest',
            cval = 0,
            channel_shift_range = 0,
            vertical_flip = False)
    mask_gen = ImageDataGenerator(
            rotation_range = 40,
            width_shift_range = 0.2,
            height_shift_range = 0.2,
            rescale = 1. / 255,
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True,
            fill_mode = 'nearest',
            cval = 0,
            channel_shift_range = 0,
            vertical_flip = False)

    image_generator = data_gen.flow_from_directory(train_path,
                                                   classes = [image_folder],
                                                   color_mode = image_color_mode,
                                                   class_mode = None,
                                                   target_size = target_size,
                                                   batch_size = batch_size,
                                                   save_to_dir = save_to_dir,
                                                   save_prefix = image_save_prefix,
                                                   seed = seed)
    mask_generator = mask_gen.flow_from_directory(train_path,
                                                  classes = [mask_folder],
                                                  class_mode = None,
                                                  color_mode = mask_color_mode,
                                                  target_size = target_size,
                                                  batch_size = batch_size,
                                                  save_to_dir = save_to_dir,
                                                  save_prefix = mask_save_prefix,
                                                  seed = seed)
    # generate image and mask(or called label)
    for i in range(0,image_num):
        image_generator.next()
        mask_generator.next()

    # adjust image and mask one by one acordding to specific rule, see adjust_img function
    image_list = []
    mask_list = []
    for file_name in os.listdir(r"./train/augmentation/"):
        if file_name[0:5]=='image':
            image_list.append(file_name)
    for image_name in image_list:
        image_last_name = image_name[5:]
        mask_name = "mask"+image_last_name
        image_file_name = './train/augmentation/' + image_name
        mask_file_name = './train/augmentation/'+ mask_name
        image = sc.imread(image_file_name)
        mask_img = sc.imread(mask_file_name)
        image,mask = adjust_img(image, mask_img)
        sc.imsave(image_file_name, image)
        sc.imsave(mask_file_name, mask)

    return

def adjust_img(img,mask):
    if np.max(img)>1:
        img = img/255
        mask = mask/255
        mask[mask>0.5] =1
        mask[mask <= 0.5] = 0
    return img, mask


# generate the images
# generate image
train_path = 'D:/PycharmProjects/U-net/train'
image_prefix = 'image'
label_prefix = 'label'
save_dir = "D:/PycharmProjects/U-net/train/augmentation"
def generate_image(batch_size,train_path,image_prefix,label_prefix, save_dir):
    train_data_augmentation(batch_size, train_path, image_prefix, label_prefix, save_to_dir = save_dir)

# define setup step
def train_set_up(train_image_path, train_label_path, train_aug_path, resize = None):
    image_list = []  # abstract the image name
    label_list = []
    for file_name in os.listdir(train_image_path):
        image_list.append(file_name)
    for file_name in os.listdir(train_label_path):
        label_list.append(file_name)
    for aug_file_name in os.listdir(train_aug_path):
        if aug_file_name[0:4] == 'mask':
            label_list.append(aug_file_name)
        else:
            image_list.append(aug_file_name)

    img_tensor_list = []
    label_tensor_list = []
    for image_name, label_name in zip(image_list, label_list):
        if os.path.exists(os.getcwd() + '\\train\\image\\' + image_name):
            img_path = os.getcwd() + '\\train\\image\\' + image_name
            label_path = os.getcwd() + '\\train\\label\\' + label_name
        else:
            img_path = os.getcwd() + '\\train\\augmentation\\' + image_name
            label_path = os.getcwd() + '\\train\\augmentation\\' + label_name

        # list to narray, then turn narray to tensor
        img = sc.imread(img_path)
        label = sc.imread(label_path)

        #  resize the image and label
        img = sc.imresize(img, resize)
        label = sc.imresize(label, resize)
        img, label = adjust_img(img, label)

        img_tensor_list.append(img)
        label_tensor_list.append(label)

    img_tensor = tf.convert_to_tensor(np.array(img_tensor_list), dtype = tf.float32)
    label_tensor = tf.convert_to_tensor(np.array(label_tensor_list), dtype = tf.float32)
    # add channel axis
    img_tensor = tf.expand_dims(img_tensor, axis = 3)
    label_tensor = tf.expand_dims(label_tensor, axis = 3)

    return img_tensor, label_tensor

# define test set up
def test_set_up(test_path, resize):
    image_list = []  # abstract the image name
    label_list = []
    for file_name in os.listdir(test_path):
        if 'predict' in file_name:
            label_list.append(file_name)
        else:
            image_list.append(file_name)

    test_image_tensor_list = []
    test_predict_tensor_list = []
    for img, pred in zip(image_list, label_list):
        img_path = os.getcwd() + '\\test\\' + img
        pred_path = os.getcwd() + '\\test\\' + pred

        test_img = sc.imread(img_path)
        test_label = sc.imread(pred_path)

        test_img = sc.imresize(test_img, resize)
        test_label = sc.imresize(test_label, resize)

        test_img, test_label = adjust_img(test_img, test_label)

        test_image_tensor_list.append(test_img)
        test_predict_tensor_list.append(test_label)

    test_image_tensor = tf.convert_to_tensor(np.array(test_image_tensor_list), dtype = tf.float32)
    test_predict_tensor = tf.convert_to_tensor(np.array(test_predict_tensor_list), dtype = tf.float32)
    # add channel axis
    test_image_tensor = tf.expand_dims(test_image_tensor, axis = 3)
    test_predict_tensor = tf.expand_dims(test_predict_tensor, axis = 3)

    return test_image_tensor, test_predict_tensor

