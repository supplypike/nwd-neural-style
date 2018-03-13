from argparse import ArgumentParser
from image import load_image
from vgg16 import load_vgg16
import numpy as np

p = ArgumentParser()
p.add_argument('image', type=str)
args = p.parse_args()

img = load_image(args.image, target_size=(224, 224))
img = np.expand_dims(img, axis=0)

with open('data/imagenet.txt') as f:
    classes = [x[:-2] for x in f.readlines()]

vgg16 = load_vgg16()
guess = vgg16.predict(img)
guess = np.argsort(-guess)

print([classes[x] for x in guess[0,:3]])
