from argparse import ArgumentParser
from image import load_image
from vgg16 import load_vgg16
import numpy as np

p = ArgumentParser()
p.add_argument('image', type=str, help='The image to classify.')
p.add_argument('-n', type=int, default=5, help='Show the top n answers.')
args = p.parse_args()

img = load_image(args.image, target_size=(224, 224))
img = np.expand_dims(img, axis=0)

with open('data/imagenet.txt') as f:
    classes = [x[:-1] for x in f.readlines()]

vgg16 = load_vgg16()
y = vgg16.predict(img)[0]
guesses = np.argsort(-y)
guesses = [classes[x] for x in guesses[:args.n]]
confidence = np.sort(y)[::-1]

for (i, guess), conf in zip(enumerate(guesses), confidence):
    print('#{}: {} ({}%)'.format(i+1, guess, int(100 * conf)))
