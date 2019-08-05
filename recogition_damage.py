# -*- coding: utf-8 -*-
from PIL import Image
import os
import pyocr
import sys
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def draw_contours(ax, img, contours):
    ax.imshow(img)  # 画像を表示する。
    ax.set_axis_off()

    for i, cnt in enumerate(contours):
        # 形状を変更する。(NumPoints, 1, 2) -> (NumPoints, 2)
        cnt = cnt.squeeze(axis=1)
        # 輪郭の点同士を結ぶ線を描画する。
        ax.add_patch(Polygon(cnt, color="b", fill=None, lw=2))
        # 輪郭の点を描画する。
        ax.plot(cnt[:, 0], cnt[:, 1], "ro", mew=0, ms=4)
        # 輪郭の番号を描画する。
        ax.text(cnt[0][0], cnt[0][1], i, color="orange", size="20")


threshold_dict = {'chara': 100, 'time': 20, 'damage': 200}
tools = pyocr.get_available_tools()

if len(tools) == 0:
    print("No OCR tool found")
    sys.exit(1)

# select pyocr.tesseract
tool = tools[0]

# make image filepath
img_filename = 'data/damage.png'
img_datatype = 'damage'      # choose from threshold_dict keys
filepath = os.path.split(__file__)[0]
url_img = os.path.join(filepath, img_filename)

# load image
img = cv2.imread(url_img)
height, width = img.shape[:2]

#plt.imshow(img, cmap='gray')
#plt.show()

# gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_thre, img_gray = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

plt.imshow(img_gray, cmap='gray')

# pick up object
contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# draw
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_title("cv2.RETR_EXTERNAL")
draw_contours(ax, img, contours)

plt.show()

# # reverse color
# rv_img = cv2.bitwise_not(img)
# plt.imshow(rv_img, cmap='gray')
# plt.show()
#
# # resize
# big_rv_img = cv2.resize(rv_img, (width*10, height*10))
# plt.imshow(big_rv_img, cmap='gray')
# plt.show()
#
# # show threshold binary
# ret, thresh1 = cv2.threshold(big_rv_img, threshold_dict[img_datatype], 255, cv2.THRESH_BINARY)
# plt.imshow(thresh1, cmap='gray')
# plt.show()
#
# # convert
# txt = tool.image_to_string(
#     Image.fromarray(thresh1),
#     builder=pyocr.builders.TextBuilder(tesseract_layout=6)
# )
# print(txt)
