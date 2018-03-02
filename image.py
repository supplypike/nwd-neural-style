from keras.preprocessing import image
import pylab
import numpy as np

AVG_PIXEL = [103.939, 116.779, 123.68]

def display_image(img):
    img = postprocess_image(img)
    pylab.imshow(image.array_to_img(img))
    pylab.pause(0.01)

def load_image(path, **kwargs):
    img = image.load_img(path, **kwargs)
    img = image.img_to_array(img)
    img = img[:,:,::-1]
    img = img - AVG_PIXEL
    return img

def save_image(array, path):
    img = postprocess_image(array)
    img = image.array_to_img(img)
    img.save(path)

def postprocess_image(array):
    y = array.copy()
    y = y + AVG_PIXEL
    y = y[:,:,::-1]
    return y

pylab.ion()
