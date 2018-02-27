from argparse import ArgumentParser
from image import display_image, load_image, save_image
from keras import backend as K
from style import apply_style
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

content_image = load_image(args.content_image, target_size=(320, 240))
style_image = load_image(args.style_image, target_size=content_image.shape)

width = content_image.shape[1]
height = content_image.shape[0]

styled = apply_style(content_image, style_image, display=args.display)
