from __future__ import print_function
from image import display_image
from latent_value import LatentValue
from keras import backend as K
from keras.layers import Concatenate, Conv2D, GlobalMaxPooling2D,  Input, MaxPooling2D
from keras.models import Model
from vgg16 import load_vgg16_weights
import keras
import numpy as np
import sys

class StyleNet():
    def __init__(self, input_shape=(224, 224, 3)):
        # Different input at the beginning

        x = Input(batch_shape=(2,)+input_shape)
        self.output_image = LatentValue(shape=(1,)+input_shape, name='output_image')(x)

        y = Concatenate(axis=0)([x, self.output_image])

        # VGG16 at the end

        y = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=False)(y)
        y = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=False)(y)
        y = MaxPooling2D(name='block1_pool')(y)

        y = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=False)(y)
        y = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=False)(y)
        y = MaxPooling2D(name='block2_pool')(y)

        y = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=False)(y)
        y = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=False)(y)
        y = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=False)(y)
        y = MaxPooling2D(name='block3_pool')(y)

        y = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=False)(y)
        y = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=False)(y)
        y = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=False)(y)
        y = MaxPooling2D(name='block4_pool')(y)

        y = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=False)(y)
        y = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=False)(y)
        y = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=False)(y)
        y = MaxPooling2D(name='block5_pool')(y)

        y = GlobalMaxPooling2D()(y)

        self.model = Model(inputs=x, outputs=y)
        load_vgg16_weights(self.model, include_top=False, by_name=True)

    def apply_style(self, content_image, style_image, display=False, **kwargs):
        K.set_value(self.output_image, np.expand_dims(content_image, axis=0))

        optimizer = keras.optimizers.sgd(lr=1e-3, momentum=0.9, nesterov=True)
        loss = self._neural_style_loss(content_image, **kwargs)

        updates = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        f = K.function(self.model.inputs, [loss], updates=updates)

        ins = np.concatenate((
            np.expand_dims(content_image, axis=0),
            np.expand_dims(style_image, axis=0)),
            axis=0)

        for i in range(100):
            print('\r{} / {}'.format(i+1, 100), end='')
            sys.stdout.flush()
            f([ins])
            if display:
                display_image(K.get_value(self.output_image[0]))
        print()

        return K.get_value(self.output_image[0])

    def _neural_style_loss(self, content_image, content_weight=0.025, tv_weight=8.5e-5, style_weight=1.):
        output = self.model.get_layer('block5_conv2').output
        content_features = output[0]
        output_features = output[2]

        content_loss = K.sum(K.square(content_features - output_features))
        tv_loss = self._total_variation_loss()

        style_loss = 0.
        conv_outputs = [x.output for x in self.model.layers if x.__class__.__name__ == 'Conv2D']
        for i in range(len(conv_outputs) - 1):
            style_features_1 = conv_outputs[i][1]
            output_features_1 = conv_outputs[i][2]
            loss_1 = self._style_loss(style_features_1, output_features_1)

            style_features_2 = conv_outputs[i+1][1]
            output_features_2 = conv_outputs[i+1][2]
            loss_2 = self._style_loss(style_features_2, output_features_2)

            style_loss = style_loss + (loss_1 - loss_2) / (2 ** (len(conv_outputs) - i - 2))

        return content_weight*content_loss + tv_weight*tv_loss + style_weight*style_loss

    def _gram_matrix(self, x):
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
        return K.dot(features - 1, K.transpose(features - 1))

    def _style_loss(self, style_features, output_features):
        S = self._gram_matrix(style_features)
        O = self._gram_matrix(output_features)
        size = int(self.output_image.shape[1] * self.output_image.shape[2])
        return K.sum(K.square(S - O)) / (36. * (size ** 2))

    def _total_variation_loss(self):
        a = K.square(self.output_image[:, :-1, :-1, :] - self.output_image[:, 1:, :-1, :])
        b = K.square(self.output_image[:, :-1, :-1, :] - self.output_image[:, :-1, 1:, :])
        return K.sum(K.pow(a + b, 1.25))
