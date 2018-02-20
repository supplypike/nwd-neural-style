from __future__ import print_function
from keras import backend as K
from vgg16 import load_vgg16

class Styler:
    def __init__(self, width, height, target_layer=16):
        self.vgg16 = load_vgg16(width, height)
        self.target_layer = self.vgg16.layers[target_layer]

    def style(self, content_image, style_image):
        width = content_image.shape[1]
        height = content_image.shape[0]

        # we feed in three images at a time to the vgg16 network:
        # 1. the content image, to preserve content after styling
        # 2. the style image, to extract style features
        # 3. the styled image, to see how well we did
        batch_input = K.concatenate((
            K.constant(content_image),
            K.constant(style_image),
            K.placeholder(content_image.shape)),
            axis=0)
