import pyocr
import cv2
import os
import sys
from PIL import Image

# read OCR tools
tools = pyocr.get_available_tools()
if len(tools) == 0:
    print("No OCR tool found")
    sys.exit(1)

# select pyocr.tesseract
tool = tools[0]

# read image file
root_dir = os.path.split(__file__)[0]

image_135_5 = os.path.join(root_dir, 'data/135_5.png')
image_6 = os.path.join(root_dir, 'data/6.png')

image = cv2.imread(image_135_5, 0)
damage_135_5 = tool.image_to_string(
    Image.fromarray(image),
    builder=pyocr.builders.TextBuilder(tesseract_layout=6)
)

image = cv2.imread(image_6, 0)
damage_6 = tool.image_to_string(
    Image.fromarray(image),
    builder=pyocr.builders.TextBuilder(tesseract_layout=6)
)

# output
print('期待値：135.5 実行結果：{}'.format(damage_135_5))
print('期待値：6     実行結果：{}'.format(damage_6))

#
# image_luci = os.path.join(root_dir, 'data/lucina.png')
# image_bayo = os.path.join(root_dir, 'data/bayonetta.png')
#
# image = cv2.imread(image_luci, 0)
# lucina = tool.image_to_string(
#     Image.fromarray(image),
#     builder=pyocr.builders.TextBuilder(tesseract_layout=6)
# )
#
# image = cv2.imread(image_bayo, 0)
# bayonetta = tool.image_to_string(
#     Image.fromarray(image),
#     builder=pyocr.builders.TextBuilder(tesseract_layout=6)
# )
#
# # output
# print('期待値：BAYONETTA 実行結果：{}'.format(bayonetta))
# print('期待値：LUCINA    実行結果：{}'.format(lucina))
