from keras import backend as K
from keras.engine.topology import Layer
from keras.preprocessing import image
import pylab
import time

def display_image(img):
    pylab.imshow(image.array_to_img(img))
    pylab.pause(0.01)

def load_image(path, **kwargs):
    img = image.load_img(path, **kwargs)
    img = image.img_to_array(img)
    return img

# a layer that ignores its input and returns an image that is trained as a weight
class Image(Layer):
    def __init__(self, init_value, **kwargs):
        super(Image, self).__init__(**kwargs)
        self.init_value = init_value.reshape((1,) + init_value.shape)

    def build(self, input_shape):
        self.value = self.add_weight(
            name='value',
            shape=self.init_value.shape,
            initializer='zeros',
            trainable=True)
        K.set_value(self.value, self.init_value)

    def call(self, x):
        return self.value

    def compute_output_shape(self, input_shape):
        return self.init_value.shape

pylab.ion()
