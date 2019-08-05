# -*- coding: utf-8 -*-
from PIL import Image
import os
import pyocr
import sys
import cv2
import matplotlib.pyplot as plt


threshold_dict = {'chara': 100, 'time': 20, 'damage': 200}
tools = pyocr.get_available_tools()

if len(tools) == 0:
    print("No OCR tool found")
    sys.exit(1)

# select pyocr.tesseract
tool = tools[0]

# make image filepath
img_filename = 'data/damage5.png'
img_datatype = 'damage'      # choose from threshold_dict keys
filepath = os.path.split(__file__)[0]
url_img = os.path.join(filepath, img_filename)

# load image
img = cv2.imread(url_img, 0)
print(type(img))
height, width = img.shape[:2]

# show gray scale
plt.imshow(img, cmap='gray')
plt.show()

# reverse color
rv_img = cv2.bitwise_not(img)
plt.imshow(rv_img, cmap='gray')
plt.show()

# resize
big_rv_img = cv2.resize(rv_img, (width*10, height*10))
plt.imshow(big_rv_img, cmap='gray')
plt.show()

# show threshold binary
ret, thresh1 = cv2.threshold(big_rv_img, threshold_dict[img_datatype], 255, cv2.THRESH_BINARY)
plt.imshow(thresh1, cmap='gray')
plt.show()

# convert
txt = tool.image_to_string(
    Image.fromarray(thresh1),
    builder=pyocr.builders.TextBuilder(tesseract_layout=6)
)
print(txt)
