import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import recogition
import pyocr
from PIL import Image

w, h = 360, 640

start = 0
end = 12000
tools = pyocr.get_available_tools()
tool = tools[0]

data_dir_path = 'E:/data3'
data2_dir_path = 'E:/data2'
file_list = os.listdir('E:/data')

print(file_list)

i = 0
for file_name in file_list:
    i = i + 1
    root, ext = os.path.splitext(file_name)
    abs_name = os.path.join(data_dir_path, file_name)
    image = cv2.imread(abs_name)
    txt = tool.image_to_string(
        Image.fromarray(image),
        builder=pyocr.builders.DigitBuilder(tesseract_layout=6)
    )
    output = os.path.join(data2_dir_path, '{}_{}.png'.format(txt, i))
    cv2.imwrite(output, image, [cv2.IMWRITE_JPEG_QUALITY, 100])

# video = cv2.VideoCapture(url_video)
# if not video.isOpened():
#     print("Could not open video")
#     sys.exit()
# ok, frame = video.read()
# if not ok:
#     print('Cannot read video file')
#     sys.exit()
#
# dmg_reco = recogition.DamageRecogition()
#
# for i in range(0, end):
#     ok, frame = video.read()
#     if(start <= i):
#         if(ok):
#             path = 'E:/data'
#             name = 'data_{}'.format(i)
#             output_path = os.path.join(path, name)
#             img_resize = cv2.resize(frame, (h, w))
#             dmg_reco.set_img(img_resize)
#             dmg_reco.write(output_path)
#
#
# video.release()
# cv2.destroyAllWindows()
