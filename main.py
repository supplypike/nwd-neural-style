from argparse import ArgumentParser
from keras.preprocessing.image import load_img
from model import make_model

p = ArgumentParser()
p.add_argument('content_image', type=str)
p.add_argument('style_image', type=str)
args = p.parse_args()

content_image = load_img(args.content_image)
style_image = load_img(args.style_image)

model = make_model(1024, 1024)
