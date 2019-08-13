import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import recogition

w, h = 360, 640

start = 0
end = 12000

fileName = 'data/sample.mp4'
filepath = os.path.split(__file__)[0]
url_video = os.path.join(filepath, fileName)

video = cv2.VideoCapture(url_video)
if not video.isOpened():
    print("Could not open video")
    sys.exit()
ok, frame = video.read()
if not ok:
    print('Cannot read video file')
    sys.exit()

dmg_reco = recogition.TestRecogition()

for i in range(0, end):
    ok, frame = video.read()
    if(start <= i):
        if(ok):
            if(i % 6 == 0):
                path = 'E:/data8'
                name = 'data_{}'.format(i)
                output_path = os.path.join(path, name)
                # img_resize = cv2.resize(frame, (h, w))
                dmg_reco.set_img(frame)
                dmg_reco.write(output_path)
                #dmg_reco.show()

        if(i % 1000 == 0):
            print(i)


video.release()
cv2.destroyAllWindows()
