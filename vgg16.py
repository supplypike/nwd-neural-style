from keras.layers import Conv2D, Dense, Flatten, GlobalMaxPooling2D, Input, MaxPooling2D
from keras.models import Model
from keras.utils.data_utils import get_file

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

def create_vgg16_topology(input_shape=(224, 224, 3), include_top=True, input_tensor=None):
    if include_top:
        if input_shape != (224, 224, 3):
            print('WARNING: Expected 224x224x3 input when using trained classifier!')
        input_shape = (224, 224, 3)

    x = Input(shape=input_shape, tensor=input_tensor)
    y = x

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

    if include_top:
        y = Flatten(name='flatten')(y)
        y = Dense(4096, activation='relu', name='fc1')(y)
        y = Dense(4096, activation='relu', name='fc2')(y)
        y = Dense(1000, activation='softmax', name='predictions')(y)
    else:
        y = GlobalMaxPooling2D()(y)

    return Model(inputs=x, outputs=y)

def load_vgg16_weights(model, include_top=True, by_name=False):
    if include_top:
        weights = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5', WEIGHTS_PATH)
    else:
        weights = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', WEIGHTS_PATH_NO_TOP)
    model.load_weights(weights, by_name=by_name)

def load_vgg16(input_shape=(224, 224, 3), include_top=True, input_tensor=None):
    model = create_vgg16_topology(input_shape, include_top, input_tensor=input_tensor)
    load_vgg16_weights(model, include_top)
    return model
