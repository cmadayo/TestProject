import os
import pickle
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import recogition

fileName = 'model.pickle'
filepath = os.path.split(__file__)[0]
uri_pickle = os.path.join(filepath, fileName)

with open(uri_pickle, mode='rb') as fp:
    lr = pickle.load(fp)





w, h = 360, 640

start = 0
end = 1000

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

time_reco = recogition.CharacterNameRecogition()

for i in range(0, end):
    ok, frame = video.read()
    if(start <= i):
        if(ok):
            # Get character name at 3rd frame (because 1st frame does not show their names)
            if(i == 2):
                print(i)
                img_resize = cv2.resize(frame, (h, w))
                time_reco.set_img(img_resize)
                txts = time_reco.get_analyzed_txts()


print(txts)

frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frame_rate = int(video.get(cv2.CAP_PROP_FPS))

video.release()
cv2.destroyAllWindows()

print(frame_count)
print(frame_rate)
