import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

def GaussianBlur(path, image):
    leaf = Image.open(path + image)
    leafmod = leaf.filter(ImageFilter.GaussianBlur(radius = 2))
    return leafmod

def ContrBright(path, image):
    image = Image.open(path + image)
    image = ImageEnhance.Contrast(image).enhance(1.5)
    image = ImageEnhance.Brightness(image).enhance(1.5)
    return image

def Reflect(path, image):
    image = Image.open(path + image)
    image = ImageOps.mirror(image)
    return image