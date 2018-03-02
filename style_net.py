from latent_value import LatentValue
from keras import backend as K
from keras.layers import Concatenate, Conv2D, GlobalMaxPooling2D,  Input, MaxPooling2D
from keras.models import Model
from vgg16 import load_vgg16_weights
import keras
import numpy as np

from image import display_image
import time

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

    def apply_style(self, content_image, style_image, **kwargs):
        K.set_value(self.output_image, np.expand_dims(content_image, axis=0))

        optimizer = keras.optimizers.sgd(lr=1e-4, momentum=0.9, nesterov=True)
        loss = self._neural_style_loss(content_image, **kwargs)

        updates = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        f = K.function(self.model.inputs, [loss], updates=updates)

        ins = np.concatenate((
            np.expand_dims(content_image, axis=0),
            np.expand_dims(style_image, axis=0)),
            axis=0)

        for i in range(100):
            print('{} / {}'.format(i+1, 100))
            f([ins])
            display_image(K.get_value(self.output_image[0]))
        time.sleep(10)

    def _neural_style_loss(self, content_image, content_weight=0.025, tv_weight=1e-4):
        output = self.model.get_layer('block5_conv2').output
        content_features = output[0]
        style_features = output[1]
        output_features = output[2]

        content_loss = content_weight * K.sum(K.square(content_image - self.output_image))
        tv_loss = tv_weight * self._total_variation_loss()
        style_loss = self._style_loss(style_features, output_features)

        return content_loss + tv_loss + style_loss

    def _gram_matrix(self, x):
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
        return K.dot(features - 1, K.transpose(features - 1))

    def _style_loss(self, style_features, output_features):
        S = self._gram_matrix(style_features)
        O = self._gram_matrix(output_features)
        size = int(style_features.shape[0] * style_features.shape[1])
        return K.sum(K.square(S - O)) / (36. * (size ** 2))

    def _total_variation_loss(self):
        a = K.square(self.output_image[:, :-1, :-1, :] - self.output_image[:, 1:, :-1, :])
        b = K.square(self.output_image[:, :-1, :-1, :] - self.output_image[:, :-1, 1:, :])
        return K.sum(K.pow(a + b, 1.25))
