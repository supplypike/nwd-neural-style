from __future__ import print_function
from keras import backend as K
from vgg16 import load_vgg16

class Styler:
    def __init__(self, width, height, feature_layer=16, content_weight=0.025):
        self.model = load_vgg16(width, height)
        self.content_weight = content_weight

        # chop off the output layers of the network
        feature_layer = self.model.layers[feature_layer]
        while self.model.layers[-1] != feature_layer:
            self.model.layers.pop()
        self.model.outputs = [feature_layer.output]
        feature_layer.outbound_nodes = []

        # freeze the weights of the network
        for layer in self.model.layers:
            layer.trainable = False

    # typical loss functions depend on labeled data,
    # but our loss only depends on the predictions,
    # which has three pieces:
    # 1. content image features
    # 2. style image features
    # 3. output iamge features
    def _loss(self, _, predictions):
        content = predictions[0, :, :, :]
        style = predictions[1, :, :, :]
        output = predictions[2, :, :, :]

        loss = self.content_weight * self._content_loss(content, output)

        return loss

    def _content_loss(self, content, output):
        return K.sum(K.square(output - content))

    def style(self, content_image, style_image):
        output_image = K.placeholder(content_image.shape)

        # we feed in three images at a time to the vgg16 network:
        # 1. the content image, to preserve content after styling
        # 2. the style image, to extract style features
        # 3. the output image, to see how well we did
        batch_input = K.concatenate((
            K.constant(content_image),
            K.constant(style_image),
            output_image),
            axis=0)

        # we provide the ...!?
        self.model.fit(batch_input, batch_input)
