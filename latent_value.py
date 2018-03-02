from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import RandomUniform

# A layer that contains a latent value learned by backpropagation.
# Due to limitations of Keras, an input tensor is required but ignored.
class LatentValue(Layer):
    def __init__(self, shape=None, init_value=None, **kwargs):
        assert(shape is not None or init_value is not None)
        assert(shape is None or init_value is None or init_value.shape == shape)

        super(LatentValue, self).__init__(**kwargs)
        self.built = True

        self.shape = shape or init_value.shape
        self.value = self.add_weight(
            name='value',
            shape=self.shape,
            initializer=RandomUniform(),
            trainable=True)
        if init_value is not None:
            K.set_value(self.value, init_value)

    def call(self, x):
        return self.value

    def compute_output_shape(self, input_shape):
        return self.shape
