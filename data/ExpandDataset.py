import numpy as np
from PIL import Image, ImageFilter
from data.ImageMod import GaussianBlur, ContrBright, Reflect
import pandas as pd

def ExpandDataset(leafdata):
    #leafdata = pd.read_csv('/content/train.csv')
    #leafdata = pd.DataFrame([['2216849948.jpg', 0]], columns=['image_id', 'label'])
    tomod = [0, 1, 2, 4]
    funcs = [GaussianBlur, ContrBright, Reflect]
    #print(leafdata)
    for i in tomod:
        lftomod = leafdata[leafdata['label']==i]
        for ind in lftomod.index:
            #print(lftomod[ind])
            #print(ind)
            imName = lftomod.loc[ind]['image_id']
            path = '/content/train_images/'
            for f in funcs:
                newImg = f(path, imName)
                newFile = path + str(i) + f.__name__ + imName
                newImg.save(newFile)
                leafdata.loc[len(leafdata.index)] = [str(i) + f.__name__ + imName, i]
            #newImGB = GaussianBlur(path, imName, i)
            #newfileGB = path + 'l' + str(i) + 'blur' + imName
            # modded_im.add(newIm)
            # modded_label.add(i)
            # leafdata.loc[len(leafdata.index)] = [imName, i]
            #newName = 'img' + str(leafdata.loc[i,'label']) + str(i) + '.jpg'
        # dicto = {'image_id':modded_im, 'label':modded_label}
        # mod_total = pd.DataFrame.from_dict(dicto)
    #print(leafdata)

