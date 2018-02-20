from argparse import ArgumentParser
from keras.preprocessing import image
from styler import Styler

import pylab
import time

def load_image(path, **kwargs):
    img = image.load_img(path, **kwargs)
    img = image.img_to_array(img)
    return img

def display_image(img):
    pylab.imshow(image.array_to_img(img))
    pylab.pause(0.01)

p = ArgumentParser()
p.add_argument('content_image', type=str)
p.add_argument('style_image', type=str)
args = p.parse_args()

content_image = load_image(args.content_image)
style_image = load_image(args.style_image, target_size=content_image.shape)

width = content_image.shape[1]
height = content_image.shape[0]

pylab.ion()

for _ in range(3):
    display_image(content_image)
    time.sleep(1)
    display_image(style_image)
    time.sleep(1)
    pylab.draw()

#styler = Styler(width, height)
#styled = styler.style(content_image, style_image)
