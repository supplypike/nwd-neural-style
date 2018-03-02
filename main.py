from __future__ import print_function
from argparse import ArgumentParser
from image import display_image, load_image, save_image
from keras import backend as K
from style_net import StyleNet
from vgg16 import load_vgg16
import numpy as np
import tensorflow as tf
import time

p = ArgumentParser()
p.add_argument('content_image', type=str)
p.add_argument('style_image', type=str)
p.add_argument('-c', '--cpu_cores', type=int, default=1)
p.add_argument('-d', '--display', action='store_true')
args = p.parse_args()

if args.cpu_cores > 0:
    config = tf.ConfigProto(
        intra_op_parallelism_threads=args.cpu_cores,
        inter_op_parallelism_threads=args.cpu_cores)
    session = tf.Session(config=config)
    K.set_session(session)

content_image = load_image(args.content_image, target_size=(240, 320))
style_image = load_image(args.style_image, target_size=content_image.shape)

style_net = StyleNet(input_shape=content_image.shape)
style_net.apply_style(content_image, style_image)
