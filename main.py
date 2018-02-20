from argparse import ArgumentParser
from image import display_image, load_image
from style import apply_style
import time

p = ArgumentParser()
p.add_argument('content_image', type=str)
p.add_argument('style_image', type=str)
args = p.parse_args()

content_image = load_image(args.content_image)
style_image = load_image(args.style_image, target_size=content_image.shape)

width = content_image.shape[1]
height = content_image.shape[0]

styled = apply_style(content_image, style_image)
