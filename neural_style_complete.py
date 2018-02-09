from __future__ import print_function

from argparse import ArgumentParser
from scipy.misc import imread, imresize

import numpy as np

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.utils.data_utils import get_file

parser = ArgumentParser(description='Neural style transfer with Keras for Nowhere Developers.')
parser.add_argument('base_image', type=str, help='Image file to which to apply style.')
parser.add_argument('style_image', type=str, help='Image file from which to transfer style.')
parser.add_argument('out_prefix', type=str, nargs='?', default='', help='Prefix for result images.')
args = parser.parse_args()

def preprocess_image(path, new_shape=None, return_info=False):
    img = imread(path, mode='RGB').astype('float32')

    r_mean = np.average(img[:, :, 0], axis=(0, 1))
    g_mean = np.average(img[:, :, 1], axis=(0, 1))
    b_mean = np.average(img[:, :, 2], axis=(0, 1))

    img[:, :, 0] -= r_mean
    img[:, :, 1] -= g_mean
    img[:, :, 2] -= b_mean

    if new_shape:
        img = imresize(img, new_shape)

    img = np.expand_dims(img, axis=0)

    if return_info:
        return img, img.shape[1], img.shape[2], (r_mean, g_mean, b_mean)
    else:
        return img

base_image, width, height, mean = preprocess_image(args.base_image, return_info=True)
style_image = preprocess_image(args.style_image, new_shape=(width, height))

base_image = K.variable(base_image)
style_image = K.variable(style_image)
output_image = K.placeholder((1, width, height, 3))
input_tensor = K.concatenate((base_image, style_image, output_image), axis=0)

x = Input(tensor=input_tensor, batch_shape=(3, width, height, 3))
y = x

y = Conv2D(64, (3, 3), activation='relu', padding='same')(y)
y = Conv2D(64, (3, 3), activation='relu', padding='same')(y)
y = MaxPooling2D()(y)

y = Conv2D(128, (3, 3), activation='relu', padding='same')(y)
y = Conv2D(128, (3, 3), activation='relu', padding='same')(y)
y = MaxPooling2D()(y)

y = Conv2D(256, (3, 3), activation='relu', padding='same')(y)
y = Conv2D(256, (3, 3), activation='relu', padding='same')(y)
y = Conv2D(256, (3, 3), activation='relu', padding='same')(y)
y = MaxPooling2D()(y)

y = Conv2D(512, (3, 3), activation='relu', padding='same')(y)
y = Conv2D(512, (3, 3), activation='relu', padding='same')(y)
y = Conv2D(512, (3, 3), activation='relu', padding='same')(y)
y = MaxPooling2D()(y)

y = Conv2D(512, (3, 3), activation='relu', padding='same')(y)
y = Conv2D(512, (3, 3), activation='relu', padding='same')(y)
y = Conv2D(512, (3, 3), activation='relu', padding='same')(y)
y = MaxPooling2D()(y)

model = Model(inputs=x, outputs=y)
weights = get_file(
    'vgg16_weights_th_dim_ordering_th_kernels_notop.h5',
    'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
)

print('Model loaded!')
