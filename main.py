from argparse import ArgumentParser
from keras.preprocessing import image
from styler import Styler

p = ArgumentParser()
p.add_argument('content_image', type=str)
p.add_argument('style_image', type=str)
args = p.parse_args()

content_image = image.img_to_array(image.load_img(args.content_image))
style_image = image.img_to_array(image.load_img(args.style_image,
    target_size=content_image.shape))

width = content_image.shape[1]
height = content_image.shape[0]

styler = Styler(width, height)
styled = styler.style(content_image, style_image)
