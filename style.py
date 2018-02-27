from __future__ import print_function
from keras import backend as K
from keras.layers import Concatenate, Input
from keras.models import Model
from image import Image, display_image, save_image
from vgg16 import load_vgg16
import keras
import numpy as np

def _gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    return K.dot(features - 1, K.transpose(features - 1))

def _content_loss(content_features, output_features):
    return K.sum(K.square(output_features - content_features))

def _style_loss(style_features, output_features):
    S = _gram_matrix(style_features)
    O = _gram_matrix(output_features)
    size = int(style_features.shape[0] * style_features.shape[1])
    return K.sum(K.square(S - O)) / (36. * (size ** 2))

def _total_variation_loss(output_image):
    a = K.square(output_image[:, :-1, :-1, :] - output_image[:, 1:, :-1, :])
    b = K.square(output_image[:, :-1, :-1, :] - output_image[:, :-1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

def _neural_style_loss(model, content_weight=0.025, tv_weight=1e-4):
    output = model.outputs[0]
    output_image = model.layers[1].output

    style_loss = 0.
    conv_layers = [x for x in model.layers[3].layers if x.__class__.__name__ == 'Conv2D']
    for i in range(len(conv_layers) - 1):
        # l1 = conv_layers[i]
        # l2 = conv_layers[i+1]
        # print(l1)
        # print(l1.output)
        # print(l1.output.shape)
        # print(l1.output[1].shape)
        # sl1 = _style_loss(l1.output[1], l2.output[2])
        # sl2 = _style_loss(l2.output[1], l2.output[2])
        # sl = sl1 - sl2
        # style_loss += sl / (2 ** (len(conv_layers) - i - 1))
        style_loss += K.sum(conv_layers[i].output)

    content_features = output[0]
    output_features = output[2]

    content_loss = content_weight * _content_loss(content_features, output_features)
    tv_loss = tv_weight * _total_variation_loss(output_image)

    return content_loss + style_loss + tv_loss

def _fit(model, ins, **kwargs):
    optimizer = keras.optimizers.sgd(lr=0.001)
    loss = _neural_style_loss(model, **kwargs)

    updates = optimizer.get_updates(params=model.trainable_weights, loss=loss)
    f = K.function(model.inputs, [loss], updates=updates)

    # todo: don't stop after just one try
    outs = f(ins)

def apply_style(content_image, style_image, display=False, content_weight=0.025, tv_weight=1e-4, feature_layer=16):
    height = content_image.shape[0]
    width = content_image.shape[1]

    # load vgg16, chop off the last layer, and freeze the weights
    vgg16 = load_vgg16(width, height, include_top=False)
    feature_layer = vgg16.layers[feature_layer]
    while feature_layer != vgg16.layers[-1]:
        vgg16.layers.pop()
    vgg16.layers[-1].outbound_nodes = []
    vgg16.outputs = [vgg16.layers[-1].output]
    for layer in vgg16.layers:
        layer.trainable = False

    # create a new model on top of vgg16 with built-in output image, compiled with custom loss
    x1 = Input(batch_shape=(2, height, width, 3))
    x2 = Image(content_image)(x1)
    y = Concatenate(axis=0)([x1, x2])
    y = vgg16(y)
    model = Model(inputs=x1, outputs=y)

    print(model.summary())
    print(vgg16.summary())

    # prepare training inputs
    ins = np.concatenate((
        np.expand_dims(content_image, axis=0),
        np.expand_dims(style_image, axis=0)),
        axis=0)

    # train to fit
    for i in range(100):
        print('{} of {}...'.format(i+1, 100))
        _fit(model, [ins], content_weight=content_weight, tv_weight=tv_weight)
        val = K.get_value(x2).reshape(x2.shape[1:])
        if display:
            display_image(val)
        else:
            save_image(val, 'img{:02}.png'.format(i+1))

    # return result
    return K.get_value(x2).reshape(x2.shape[1:])
