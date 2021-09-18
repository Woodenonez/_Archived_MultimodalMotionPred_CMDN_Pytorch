import os, sys

import numpy as np

from PIL import Image

# img_np = np.array([[0,0,0],[255,255,255],[0,0,0]]).astype(np.uint8)
# img = Image.fromarray(img_np)
# img.save('test.bmp')

img = Image.open('test.bmp')
print(np.array(img))