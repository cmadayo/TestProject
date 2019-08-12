import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import recogition
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

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

time_reco = recogition.CharacterNameRecogition()
test_reco = recogition.TestRecogition()

plot_data_p1 = [0]
plot_data_p2 = [0]
plot_data_time = [0]


for i in range(0, end):
    ok, frame = video.read()
    if(start <= i):
        if(ok):
            # Get character name at 3rd frame (because 1st frame does not show their names)
            img_resize = cv2.resize(frame, (h, w))
            if(i == start + 2):
                print(i)
                time_reco.set_img(img_resize)
                txts = time_reco.get_analyzed_txts()

            test_reco.set_img(img_resize)
            damages = test_reco.get_analyzed_txts()
            # plot_data_p1 = np.append(plot_data_p1, int(damages[0]))
            # plot_data_p2 = np.append(plot_data_p1, int(damages[1]))
            # plot_data_time = np.append(plot_data_time, i)
            if(int(damages[0]) == 10):
                plot_data_p1.append(plot_data_p1[-1])
            else:
                plot_data_p1.append(int(damages[0]))
            if(int(damages[1]) == 10):
                plot_data_p2.append(plot_data_p1[-1])
            else:
                plot_data_p2.append(int(damages[1]))

            plot_data_time.append(i)

plt.plot(plot_data_time, plot_data_p1)
plt.plot(plot_data_time, plot_data_p2)
plt.show()
print(txts)

video.release()
cv2.destroyAllWindows()
