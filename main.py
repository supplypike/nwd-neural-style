from __future__ import print_function
from argparse import ArgumentParser
from image import display_image, load_image, save_image
from keras import backend as K
from style_net import StyleNet
import numpy as np
import pylab
import tensorflow as tf

p = ArgumentParser()
p.add_argument('content_image', type=str)
p.add_argument('style_image', type=str)
p.add_argument('-c', '--cpu_cores', type=int, default=1, help='The number of cores to use when running on a CPU.')
p.add_argument('-s', '--size', type=int, default=None, help='The maximum length of dimension, for resizing the input images.')
p.add_argument('--no_display', dest='display', action='store_false')
args = p.parse_args()

if args.cpu_cores > 0:
    config = tf.ConfigProto(
        intra_op_parallelism_threads=args.cpu_cores,
        inter_op_parallelism_threads=args.cpu_cores)
    session = tf.Session(config=config)
    K.set_session(session)

content_image = load_image(args.content_image, max_size=args.size)
style_image = load_image(args.style_image, target_size=content_image.shape[:-1])

style_net = StyleNet(input_shape=content_image.shape)
styled_image = style_net.apply_style(content_image, style_image, display=args.display)
display_image(styled_image)

pylab.ioff()
pylab.show()
