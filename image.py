from keras.preprocessing import image
import pylab
import numpy as np
from PIL import Image

AVG_PIXEL = [103.939, 116.779, 123.68]

def display_image(img):
    img = postprocess_image(img)
    pylab.imshow(image.array_to_img(img))
    pylab.pause(0.01)

def load_image(path, target_size=None):
    img = image.load_img(path)
    if target_size:
        # change target size from row-major to column-major
        img = img.resize(target_size[::-1], resample=Image.BILINEAR)
    img = image.img_to_array(img)
    img = img[...,::-1]
    img = img - AVG_PIXEL
    return img

def save_image(array, path):
    img = postprocess_image(array)
    img = image.array_to_img(img)
    img.save(path)

def postprocess_image(array):
    y = array.copy()
    y = y + AVG_PIXEL
    y = y[...,::-1]
    y = np.clip(y, 0, 255).astype('uint8')
    return y

pylab.ion()
