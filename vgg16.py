from keras.layers import Conv2D, Input, MaxPooling2D
from keras.models import Model
from keras.utils.data_utils import get_file

def load_vgg16(width, height, channels=3, **kwargs):
    x = Input(shape=(width, height, channels))
    y = x

    y = Conv2D(64, (3, 3), activation='relu', **kwargs)(y)
    y = Conv2D(64, (3, 3), activation='relu', **kwargs)(y)
    y = MaxPooling2D()(y)

    y = Conv2D(128, (3, 3), activation='relu', **kwargs)(y)
    y = Conv2D(128, (3, 3), activation='relu', **kwargs)(y)
    y = MaxPooling2D()(y)

    y = Conv2D(256, (3, 3), activation='relu', **kwargs)(y)
    y = Conv2D(256, (3, 3), activation='relu', **kwargs)(y)
    y = Conv2D(256, (3, 3), activation='relu', **kwargs)(y)
    y = MaxPooling2D()(y)

    y = Conv2D(512, (3, 3), activation='relu', **kwargs)(y)
    y = Conv2D(512, (3, 3), activation='relu', **kwargs)(y)
    y = Conv2D(512, (3, 3), activation='relu', **kwargs)(y)
    y = MaxPooling2D()(y)

    y = Conv2D(512, (3, 3), activation='relu', **kwargs)(y)
    y = Conv2D(512, (3, 3), activation='relu', **kwargs)(y)
    y = Conv2D(512, (3, 3), activation='relu', **kwargs)(y)
    y = MaxPooling2D()(y)

    model = Model(inputs=x, outputs=y)
    weights = get_file(
        'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
        'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    )
    model.load_weights(weights)

    return model
