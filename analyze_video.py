import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import recognition
from typing import Tuple

#########################
# SETTING
#########################
start_index = 0                         # index when start analyzing video(frame count) TODO: combine
end_index = 12000                       # index when end analyzing video(frame count) TODO: combine
video_filepath = 'data/sample.mp4'      # video file name TODO: combine
progress_step = 100                     # step count for debug log
recognize_step = 30                     # step count for recoginizing(30 frame = 0.5s)
#########################


def update_damage_for_plot_data(damage: int, last_damage: int, last_plot: int) -> Tuple[int, int]:
    if(damage == -1):                   # negative value if analysis failed
        return(last_plot, last_plot)

    elif(damage != last_damage):        # Because damage change is unstable, it is temporarily suspended
        return(last_plot, damage)

    else:
        return(damage, damage)


# make video file's uri
root_dir = os.path.split(__file__)[0]
video_uri = os.path.join(root_dir, video_filepath)

# video capture start and check if the file was opened
video = cv2.VideoCapture(video_uri)
if not video.isOpened():
    print("Could not open video")
    sys.exit()

# check if the file can be read
ok, frame = video.read()
if not ok:
    print('Cannot read video file')
    sys.exit()

# initialize string recogition instance
cnr = recognition.CharacterNameRecogition()
dr = recognition.DamageRecogition()

# initialize data for plot
plot_list_p1 = [0]           # player1 damage data
plot_list_p2 = [0]           # player2 damage data
plot_list_time = [0]         # time data(seconds)
last_dmg_p1 = 0              # last player1 damage data for plot algorithm
last_dmg_p2 = 0              # last player2 damage data for plot algorithm

# start reading video
for i in range(0, end_index):
    ok, frame = video.read()
    if(start_index <= i):
        if(i % progress_step == 0):
            print('Read progress rate {}/{}'.format(i-start_index, end_index-start_index))

        if(ok):
            # Get character name at 3rd frame (because 1st frame does not show their names)
            if(i == start_index + 2):
                cnr.set_img(frame)
                character_names = cnr.get_recognized_texts()

            # recognize damage every 0.5s(30 frames)
            if(i % recognize_step == 0):
                dr.set_img(frame)
                dmg_p1, dmg_p2 = dr.get_recognized_texts()
                dmg_p1, dmg_p2 = (int(dmg_p1), int(dmg_p2))

                # update value for plot data
                dmg_p1, last_dmg_p1 = update_damage_for_plot_data(dmg_p1, last_dmg_p1, plot_list_p1[-1])
                dmg_p2, last_dmg_p2 = update_damage_for_plot_data(dmg_p2, last_dmg_p2, plot_list_p2[-1])

                # append values for plot
                plot_list_p1.append(dmg_p1)
                plot_list_p2.append(dmg_p2)
                plot_list_time.append(i/60)

# release resource
video.release()
cv2.destroyAllWindows()
print('Read complete!!')

# plot data convert to numpy array
x = np.array(plot_list_time)
y_p1 = np.array(plot_list_p1)
y_p2 = np.array(plot_list_p2)

# plot configuring
fig, ax = plt.subplots()

ax.plot(x, y_p1, label=character_names[0], color='#CE749C')
ax.plot(x, y_p2, label=character_names[1], color='#49AAD2')

major_ticks_hor = np.arange(0, np.amax(x), 60)
minor_ticks_hor = np.arange(0, np.amax(x), 0.5)

major_ticks_ver = np.arange(0, max(np.amax(y_p1), np.amax(y_p2)), 10)
minor_ticks_ver = np.arange(0, max(np.amax(y_p1), np.amax(y_p2)), 1)

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

# plot show
plt.show()
