from __future__ import print_function
from image import display_image
from latent_value import LatentValue
from keras import backend as K
from keras.layers import Concatenate, Conv2D, GlobalMaxPooling2D,  Input, MaxPooling2D
from keras.models import Model
from vgg16 import load_vgg16
import keras
import numpy as np
import sys

class StyleNet():
    def __init__(self,
                 content_layer='block4_conv2',
                 style_layers=['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']):
        self.content_layer = content_layer
        self.style_layers = style_layers
        self.model = load_vgg16((None, None, 3), include_top=False)

    def apply_style(self, content_image, style_image, display=False, iterations=1000, content_weight=5., style_weight=500., variation_weight=100.):
        content_outputs = [x.output for x in self.model.layers if x.name == self.content_layer]
        content_func = K.function(self.model.inputs, content_outputs)
        content_features = content_func([np.expand_dims(content_image, axis=0)])

        style_outputs = [x.output for x in self.model.layers if x.name in self.style_layers]
        style_func = K.function(self.model.inputs, style_outputs)
        style_features = style_func([np.expand_dims(style_image, axis=0)])
        style_features = [self._gram_matrix(x[0]) for x in style_features]

        generated_image = K.variable(np.expand_dims(content_image, axis=0))
        generated_content_features = Model(inputs=self.model.input, outputs=content_outputs)(generated_image)
        generated_style_features = Model(inputs=self.model.input, outputs=style_outputs)(generated_image)

        content_loss = content_weight * self._content_loss(content_features, generated_content_features)
        style_loss = style_weight * self._style_loss(style_features, generated_style_features)
        variation_loss = variation_weight * self._variation_loss(generated_image)
        total_loss = content_loss + style_loss + variation_loss

        optimizer = keras.optimizers.adam(lr=1.)
        updates = optimizer.get_updates(loss=total_loss, params=[generated_image])
        f = K.function(self.model.inputs, [content_loss, style_loss, variation_loss], updates=updates)

        for i in range(iterations):
            print('{} / {}'.format(i+1, iterations), end='')
            sys.stdout.flush()

            gen = K.get_value(generated_image)
            if display:
                display_image(gen[0])

            losses = f([gen])
            print(', content_loss={}, style_loss={}, variation_loss={}'.format(losses[0], losses[1], losses[2]))
            sys.stdout.flush()
        print()

        return K.get_value(generated_image)

    def _content_loss(self, content_features, generated_features):
        return K.mean(K.square(content_features[0] - generated_features[0]))

    def _style_loss(self, style_features, generated_features):
        loss = 0.
        for style_feat, gen_feat in zip(style_features, generated_features):
            loss = loss + K.mean(K.square(style_feat - self._gram_matrix(gen_feat[0])))
        return loss

    def _variation_loss(self, generated_image):
        a = K.sum(K.square(generated_image[:, :-1, :, :] - generated_image[:, 1:, :, :]))
        b = K.sum(K.square(generated_image[:, :, :-1, :] - generated_image[:, :, 1:, :]))
        return a + b

    def _gram_matrix(self, x):
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
        shape = K.shape(x)
        return K.dot(features, K.transpose(features)) / K.cast(shape[0] * shape[1], x.dtype)
