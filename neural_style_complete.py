from __future__ import print_function

from argparse import ArgumentParser
from scipy.misc import imread, imresize, imsave
from scipy.optimize import fmin_l_bfgs_b

import numpy as np

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.utils.data_utils import get_file

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

def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    return K.dot(features - 1, K.transpose(features - 1))

def style_loss(style, output):
    S = gram_matrix(style)
    C = gram_matrix(output)
    return K.sum(K.square(S - C)) / (36. * ((width * height) ** 2))

def content_loss(base, output):
    return K.sum(K.square(output - base))

def total_variation_loss(x):
    a = K.square(x[:, :width - 1, :height - 1, :] - x[:, 1:, :height - 1, :])
    b = K.square(x[:, :width - 1, :height - 1, :] - x[:, :width - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

def make_conv(filters, x, shape=(3, 3)):
    return Conv2D(filters, shape, activation='relu', padding='same')(x)

class Evaluator:
    def loss(self, x):
        x = x.reshape(1, width, height, 3)
        outs = outputs([x])
        self.loss_value = outs[0]
        self.grad_values = np.array(outs[1:]).flatten()
        return self.loss_value

    def grads(self, x):
        return self.grad_values

parser = ArgumentParser(description='Neural style transfer with Keras for Nowhere Developers.')
parser.add_argument('base_image', type=str, help='Image file to which to apply style.')
parser.add_argument('style_image', type=str, help='Image file from which to transfer style.')
parser.add_argument('out_prefix', type=str, nargs='?', default='', help='Prefix for result images.')
parser.add_argument('--content_layer', type=int, default=-2)
parser.add_argument('--content_weight', type=float, default=0.025)
parser.add_argument('--num_iter', type=int, default=10)
parser.add_argument('--style_weight', type=float, default=1)
parser.add_argument('--total_variation_weight', type=float, default=1e-4)
args = parser.parse_args()

base_image, width, height, mean = preprocess_image(args.base_image, return_info=True)
style_image = preprocess_image(args.style_image, new_shape=(width, height))

base_image = K.variable(base_image)
style_image = K.variable(style_image)
output_image = K.placeholder((1, width, height, 3))
input_tensor = K.concatenate((base_image, style_image, output_image), axis=0)

x = Input(tensor=input_tensor, batch_shape=(3, width, height, 3))
y = x

y = make_conv(64, y)
y = make_conv(64, y)
y = MaxPooling2D()(y)

y = make_conv(128, y)
y = make_conv(128, y)
y = MaxPooling2D()(y)

y = make_conv(256, y)
y = make_conv(256, y)
y = make_conv(256, y)
y = MaxPooling2D()(y)

y = make_conv(512, y)
y = make_conv(512, y)
y = make_conv(512, y)
y = MaxPooling2D()(y)

y = make_conv(512, y)
y = make_conv(512, y)
y = make_conv(512, y)
y = MaxPooling2D()(y)

model = Model(inputs=x, outputs=y)
weights = get_file(
    'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
    'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
)
model.load_weights(weights)

print('Model loaded!')

conv_layers = [1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17]
conv_layers = [model.layers[i] for i in conv_layers]

base_features = conv_layers[args.content_layer].output[0, :, :, :]
output_features = conv_layers[args.content_layer].output[2, :, :, :]
loss = args.content_weight * content_loss(base_features, output_features)

for i in range(len(conv_layers) - 1):
    style_features = conv_layers[i].output[1, :, :, :]
    output_features = conv_layers[i].output[2, :, :, :]
    sl1 = style_loss(style_features, output_features)

    shape = conv_layers[i + 1].output_shape
    style_features = conv_layers[i + 1].output[1, :, :, :]
    output_features = conv_layers[i + 1].output[2, :, :, :]
    sl2 = style_loss(style_features, output_features)

    loss += args.style_weight * (sl1 - sl2) / (2 ** (len(conv_layers) - i - 2))

loss += args.total_variation_weight * total_variation_loss(output_image)
grads = K.gradients(loss, output_image)
outputs = K.function([output_image], [loss] + grads)

current_output = preprocess_image(args.base_image)

print('Beginning training!')
evaluator = Evaluator()
prev_min_val = -1

for i in range(args.num_iter):
    print('Iteration {} / {}'.format(i + 1, args.num_iter))

    current_output, min_val, info = fmin_l_bfgs_b(
        evaluator.loss,
        current_output.flatten(),
        fprime=evaluator.grads,
        maxfun=20
    )

    if prev_min_val == -1:
        prev_min_val = min_val

    improvement = (prev_min_val - min_val) / prev_min_val * 100
    print('Loss: {}, Improvement: {}'.format(min_val, improvement))

    img = postprocess_image(current_output.copy())
    imsave(args.out_prefix + '_{}.png'.format(i))
