import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import recogition
import pyocr
from PIL import Image
import pickle

w, h = 360, 640

start = 0
end = 12000

data_dir_path = 'E:/data8'
data2_dir_path = 'E:/data9'
file_list = os.listdir(data_dir_path)

test_reco = recogition.TestRecogition()

fileName = 'model3.pickle'
filepath = os.path.split(__file__)[0]
uri_pickle = os.path.join(filepath, fileName)

with open(uri_pickle, mode='rb') as fp:
    model = pickle.load(fp)

i = 0
for file_name in file_list:
    i = i + 1
    root, ext = os.path.splitext(file_name)
    abs_name = os.path.join(data_dir_path, file_name)
    image_org = cv2.imread(abs_name, 0)
    # image = cv2.resize(image_org, (36, 50))

    # orgHeight, orgWidth = image.shape[:2]
    # print(orgHeight, orgWidth)
    # size = (int(orgWidth/5), int(orgHeight/5))   # 36*50
    # print(size)
    # image = cv2.resize(image, size)
    number = str(model.predict(image_org.reshape(1, -1))[0])

    output_dir = os.path.join(data2_dir_path, number)
    output_filename = os.path.join(output_dir, '{}.png'.format(i))

    cv2.imwrite(output_filename, image_org, [cv2.IMWRITE_JPEG_QUALITY, 100])
    if(i % 1000 == 0):
        print(i)

    # cv2.imwrite(output, image, [cv2.IMWRITE_JPEG_QUALITY, 100])

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
