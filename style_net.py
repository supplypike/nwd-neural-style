from latent_value import LatentValue
from keras import backend as K
from keras.layers import Concatenate, Conv2D, GlobalMaxPooling2D,  Input, MaxPooling2D
from keras.models import Model
from vgg16 import load_vgg16_weights
import numpy as np

class StyleNet():
    def __init__(self, input_shape=(224, 224, 3)):
        # Different input at the beginning

        x = Input(batch_shape=(2,)+input_shape)
        self.output_image = LatentValue(shape=(1,)+input_shape, name='output_image')(x)

        y = Concatenate(axis=0)([x, self.output_image])

        # VGG16 at the end

        y = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(y)
        y = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(y)
        y = MaxPooling2D(name='block1_pool')(y)

        y = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(y)
        y = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(y)
        y = MaxPooling2D(name='block2_pool')(y)

        y = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(y)
        y = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(y)
        y = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(y)
        y = MaxPooling2D(name='block3_pool')(y)

        y = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(y)
        y = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(y)
        y = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(y)
        y = MaxPooling2D(name='block4_pool')(y)

        y = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(y)
        y = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(y)
        y = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(y)
        y = MaxPooling2D(name='block5_pool')(y)

        y = GlobalMaxPooling2D()(y)

        self.model = Model(inputs=x, outputs=y)
        load_vgg16_weights(self.model, include_top=False, by_name=True)

    def apply_style(self, content_image, style_image):
        K.set_value(self.output_image, np.expand_dims(content_image, axis=0))
