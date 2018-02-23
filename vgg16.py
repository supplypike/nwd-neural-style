from keras.layers import Conv2D, GlobalMaxPooling2D, Input, MaxPooling2D
from keras.models import Model
from keras.utils.data_utils import get_file

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

def load_vgg16(width, height, include_top=True):
    x = Input(shape=(width, height, 3))
    y = x

    y = Conv2D(64, (3, 3), activation='relu')(y)
    y = Conv2D(64, (3, 3), activation='relu')(y)
    y = MaxPooling2D()(y)

    y = Conv2D(128, (3, 3), activation='relu')(y)
    y = Conv2D(128, (3, 3), activation='relu')(y)
    y = MaxPooling2D()(y)

    y = Conv2D(256, (3, 3), activation='relu')(y)
    y = Conv2D(256, (3, 3), activation='relu')(y)
    y = Conv2D(256, (3, 3), activation='relu')(y)
    y = MaxPooling2D()(y)

    y = Conv2D(512, (3, 3), activation='relu')(y)
    y = Conv2D(512, (3, 3), activation='relu')(y)
    y = Conv2D(512, (3, 3), activation='relu')(y)
    y = MaxPooling2D()(y)

    y = Conv2D(512, (3, 3), activation='relu')(y)
    y = Conv2D(512, (3, 3), activation='relu')(y)
    y = Conv2D(512, (3, 3), activation='relu')(y)
    y = MaxPooling2D()(y)

    if include_top:
        y = Flatten(name='flatten')(y)
        y = Dense(4096, activation='relu')(y)
        y = Dense(4096, activation='relu')(y)
        y = Dense(1000, activation='softmax')(y)
    else:
        y = GlobalMaxPooling2D()(y)

    model = Model(inputs=x, outputs=y)
    if include_top:
        weights = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5', WEIGHTS_PATH)
    else:
        weights = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', WEIGHTS_PATH_NO_TOP)
    model.load_weights(weights)

    return model
