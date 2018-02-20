from argparse import ArgumentParser
from image import display_image, load_image
from styler import Styler
import time

p = ArgumentParser()
p.add_argument('content_image', type=str)
p.add_argument('style_image', type=str)
args = p.parse_args()

content_image = load_image(args.content_image)
style_image = load_image(args.style_image, target_size=content_image.shape)

width = content_image.shape[1]
height = content_image.shape[0]

for _ in range(3):
    display_image(content_image)
    time.sleep(1)
    display_image(style_image)
    time.sleep(1)

#styler = Styler(width, height)
#styled = styler.style(content_image, style_image)
