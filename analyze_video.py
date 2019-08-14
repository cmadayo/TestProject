import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import recogition
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

h, w = 360, 640

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

last_damage_p1 = 0
last_damage_p2 = 0
for i in range(0, end):
    ok, frame = video.read()
    if(start <= i):
        if(ok):
            if(i%100 == 0):
                print(i)

            # Get character name at 3rd frame (because 1st frame does not show their names)
            # img_resize = cv2.resize(frame, (w, h))
            if(i == start + 2):
                print(i)
                # time_reco.set_img(img_resize)
                time_reco.set_img(frame)
                character_names = time_reco.get_analyzed_txts()

            if(i%30 == 0):
                # test_reco.set_img(img_resize)
                test_reco.set_img(frame)
                damages = test_reco.get_analyzed_txts()
            # plot_data_p1 = np.append(plot_data_p1, int(damages[0]))
            # plot_data_p2 = np.append(plot_data_p1, int(damages[1]))
            # plot_data_time = np.append(plot_data_time, i)
                if(int(damages[0]) == -1):
                    damage_p1 = plot_data_p1[-1]
                    last_damage_p1 = plot_data_p1[-1]
                elif(int(damages[0]) != last_damage_p1):
                    damage_p1 = plot_data_p1[-1]
                    last_damage_p1 = int(damages[0])
                else:
                    damage_p1 = int(damages[0])
                    last_damage_p1 = int(damages[0])

                if(int(damages[1]) == -1):
                    damage_p2 = plot_data_p2[-1]
                    last_damage_p2 = plot_data_p2[-1]
                    # print('<Pattern1> damage_p2={}, last_damage_p2={}'.format(damage_p2, last_damage_p2))
                elif(int(damages[1]) != last_damage_p2):
                    damage_p2 = plot_data_p2[-1]
                    last_damage_p2 = int(damages[1])
                    # print('<Pattern2> damage_p2={}, last_damage_p2={}'.format(damage_p2, last_damage_p2))
                else:
                    damage_p2 = int(damages[1])
                    last_damage_p2 = int(damages[1])
                    # print('<Pattern3> damage_p2={}, last_damage_p2={}'.format(damage_p2, last_damage_p2))

                # if(damage_p1 != plot_data_p1[-1] or damage_p2 != plot_data_p2[-1]):
                # for removing noise data
                # if(damage_p1 == last_damage_p1 or damage_p2 == plot_data_p2[-1]):
                plot_data_p1.append(damage_p1)
                plot_data_p2.append(damage_p2)
                plot_data_time.append(i/60)

                last_damage_p1 = last_damage_p1
                last_damage_p2 = last_damage_p2


y = np.array(plot_data_time)
X_p1 = np.array(plot_data_p1)
X_p2 = np.array(plot_data_p2)
fig, ax = plt.subplots()

# ax.stem(np.array(plot_data_time)-0.15, plot_data_p1, label=character_names[0], linefmt=' ', markerfmt='C6-', basefmt='C7')
# ax.stem(np.array(plot_data_time)+0.15, plot_data_p2, label=character_names[1], linefmt=' ', markerfmt='C9-', basefmt='C7')
# ax.stem(y, X_p1, label=character_names[0], linefmt=' ', markerfmt='C6-', basefmt='C7')
# ax.stem(y, X_p2, label=character_names[1], linefmt=' ', markerfmt='C9-', basefmt='C7')

ax.plot(y, X_p1, label=character_names[0], color='#CE749C')
ax.plot(y, X_p2, label=character_names[1], color='#49AAD2')

major_ticks_hor = np.arange(0, np.amax(y), 60)
minor_ticks_hor = np.arange(0, np.amax(y), 0.5)

major_ticks_ver = np.arange(0, max(np.amax(X_p1), np.amax(X_p2)), 10)
minor_ticks_ver = np.arange(0, max(np.amax(X_p1), np.amax(X_p2)), 1)

ax.set_xticks(major_ticks_hor)
ax.set_xticks(minor_ticks_hor, minor=True)
ax.set_yticks(major_ticks_ver)
ax.set_yticks(minor_ticks_ver, minor=True)

ax.grid(which='both')
ax.grid(which='minor', alpha=0.4)
ax.grid(which='major', alpha=0.7)

ax.legend()
ax.set_title('Smalyzer Graph')
ax.set_xlabel('Time(s)')
ax.set_ylabel('Damage(%)')
plt.show()

# print(plot_data_p1)
# plt.plot(plot_data_time, plot_data_p1)
# plt.plot(plot_data_time, plot_data_p2)
# plt.show()
# print(txts)

video.release()
cv2.destroyAllWindows()
