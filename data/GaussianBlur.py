'''
    PIC 16A Homework 5
    Author: Gabriel Sison
    UID: 705342759
    Discussion 3A
    Date: 2023-02-25
'''

import numpy as np
from PIL import Image, ImageFilter

def test_on_image():
    filter = np.ones((11,11))*(1.0/11**2)
    leaf = Image.open('/content/test_images/2216849948.jpg')
    leafmod = leaf.filter(ImageFilter.GaussianBlur(radius = 2))
    
    leafmod.show()
    leafmod.save('/content/test_images/test_image.jpg')

def GaussianBlur(path, image, label):
    #filter = np.ones((11,11))*(1.0/11**2)
    leaf = Image.open(path + image)
    leafmod = leaf.filter(ImageFilter.GaussianBlur(radius = 2))
    leafmod.show()
    #leafmod.save(path + label + image)
    return leafmod
