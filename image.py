from keras.preprocessing import image
import pylab
import time

def display_image(img):
    pylab.imshow(image.array_to_img(img))
    pylab.pause(0.01)

def load_image(path, **kwargs):
    img = image.load_img(path, **kwargs)
    img = image.img_to_array(img)
    return img

pylab.ion()
