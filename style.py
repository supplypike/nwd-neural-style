from __future__ import print_function
from keras import backend as K
from keras.layers import Concatenate, Input
from keras.models import Model
from image import Image
from vgg16 import load_vgg16
import numpy as np

def _neural_style_loss():
    pass

def apply_style(content_image, style_image, content_weight=0.025, feature_layer=16):
    height = content_image.shape[0]
    width = content_image.shape[1]

    # load vgg16, chop off the last layer, and freeze the weights
    vgg16 = load_vgg16(width, height, include_top=False)
    feature_layer = vgg16.layers[feature_layer]
    while feature_layer != vgg16.layers[-1]:
        vgg16.layers.pop()
    for layer in vgg16.layers:
        layer.trainable = False

    # create a new model on top of vgg16 with built-in output image, compiled with custom loss
    x1 = Input(batch_shape=(2, height, width, 3))
    x2 = Image(content_image)(x1)
    y = Concatenate(axis=0)([x1, x2])
    y = vgg16(y)
    model = Model(inputs=x1, outputs=y)
    model.compile(loss=_neural_style_loss, optimizer='sgd')

    # train to fit
    input_batch = K.concatenate((
        np.expand_dims(content_image, axis=0),
        np.expand_dims(style_image, axis=0)),
        axis=0)
    model.fit(x=input_batch, y=input_batch)

    # return result
    return K.get_value(output.value)
