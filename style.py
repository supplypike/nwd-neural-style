from __future__ import print_function
from keras import backend as K
from keras.layers import Concatenate, Input
from keras.models import Model
from image import Image, display_image, save_image
from vgg16 import load_vgg16
import keras
import numpy as np

def _neural_style_loss(output):
    content_features = output[0][0]
    style_features = output[0][1]
    output_features = output[0][2]

    #content_loss = K.sum(K.square(output_features - content_features))
    content_loss = K.sum(K.square(output_features - style_features))
    style_loss = 0.

    return content_loss + style_loss

def _fit(model, ins):
    optimizer = keras.optimizers.sgd()
    loss = _neural_style_loss(model.outputs)

    updates = optimizer.get_updates(params=model.trainable_weights, loss=loss)
    f = K.function(model.inputs, [loss], updates=updates)

    # todo: don't stop after just one try
    outs = f(ins)

def apply_style(content_image, style_image, display=False, content_weight=0.025, feature_layer=16):
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

    # prepare training inputs
    ins = np.concatenate((
        np.expand_dims(content_image, axis=0),
        np.expand_dims(style_image, axis=0)),
        axis=0)

    # train to fit
    for i in range(100):
        print('{} of {}...'.format(i+1, 100))
        _fit(model, [ins])
        if display:
            display_image(styled)
        else:
            val = K.get_value(x2).reshape(x2.shape[1:])
            save_image(val, 'img{:02}.png'.format(i+1))

    # return result
    return K.get_value(x2).reshape(x2.shape[1:])
