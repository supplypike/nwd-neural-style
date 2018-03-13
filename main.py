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
p.add_argument('-d', '--display', action='store_true')
p.add_argument('-i', '--iterations', type=int, default=1000, help='The number of iterations to run.')
p.add_argument('-o', '--output_image', type=str, default=None, help='(Optional) Where to save the styled image.')
p.add_argument('-s', '--size', type=int, default=None, help='The maximum length of dimension, for resizing the input images.')
p.add_argument('--content_weight', type=float, default=5., help='The weight on the content loss.')
p.add_argument('--style_weight', type=float, default=500., help='The weight on the style loss.')
p.add_argument('--variation_weight', type=float, default=100., help='The weight on the variation loss.')
args = p.parse_args()

if args.cpu_cores > 0:
    config = tf.ConfigProto(
        intra_op_parallelism_threads=args.cpu_cores,
        inter_op_parallelism_threads=args.cpu_cores)
    session = tf.Session(config=config)
    K.set_session(session)

if args.display:
    pylab.ion()

if not args.display and not args.output_image:
    print('WARNING: You have not turned on display or provided an output image path, so you will NOT see the styled results.')

content_image = load_image(args.content_image, max_size=args.size)
style_image = load_image(args.style_image, max_size=args.size)

style_net = StyleNet()
styled_image = style_net.apply_style(content_image, style_image,
                                     display=args.display,
                                     iterations=args.iterations,
                                     content_weight=args.content_weight,
                                     style_weight=args.style_weight,
                                     variation_weight=args.variation_weight)

if args.output_image:
    save_image(styled_image, args.output_image)

if args.display:
    display_image(styled_image)
    pylab.ioff()
    pylab.show()
