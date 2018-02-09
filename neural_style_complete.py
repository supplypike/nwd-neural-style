from __future__ import print_function

from argparse import ArgumentParser
from scipy.misc import imread

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

def preprocess_image(path, return_info=False):
    img = imread(path, mode='RGB').astype('float32')

    r_mean = np.average(img[:, :, 0], axis=(0, 1))
    g_mean = np.average(img[:, :, 1], axis=(0, 1))
    b_mean = np.average(img[:, :, 2], axis=(0, 1))

    img[:, :, 0] -= r_mean
    img[:, :, 1] -= g_mean
    img[:, :, 2] -= b_mean

    if return_info:
        return img, img.shape[0], img.shape[1], (r_mean, g_mean, b_mean)
    else:
        return img

base_image, width, height, mean = preprocess_image(args.base_image, return_info=True)
style_image = preprocess_image(args.style_image)
