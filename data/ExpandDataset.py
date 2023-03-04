import numpy as np
from PIL import Image, ImageFilter
from data.gaussianblur import GaussianBlur
import pandas as pd

def ExpandDataset():
    #leafdata = pd.read_csv('/content/train.csv')
    leafdata = pd.DataFrame([['2216849948.jpg', 0]], columns=['image_id', 'label'])
    tomod = [0, 1, 2, 4]
    totaldf = pd.DataFrame()

    for i in tomod:
        lftomod = leafdata[leafdata['label']==i]
        modded = []
        for data in lftomod:
            imName = data['image_id']
            path = '/content/test_images/'
            newIm = GaussianBlur(path, imName, i)
            newIm.show()
            newIm.save(path + 'l' + i + 'blur' + imName)
