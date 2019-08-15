import cv2
import sys
import os
import recognition

#########################
# SETTING
#########################
start_index = 0                                          # index when start analyzing video(frame count)
end_index = 12000                                        # index when end analyzing video(frame count)
video_filepath = 'data/sample.mp4'                       # video file name
progress_step = 100                                      # step count for debug log
write_step = 6                                           # step count for write
write_mode = 1                                           # 0: write all, 1: write only damage, 2: write only character_name
output_dir = 'E:/develop/test/sampledata'                # output directory
#########################

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

for i in range(0, end_index):
    ok, frame = video.read()
    if(start_index <= i):
        if(i % progress_step == 0):
            print('Read progress rate {}/{}'.format(i-start_index, end_index-start_index))

        if(ok):
            if(i % write_step == 0):
                if(write_mode == 0 or write_mode == 2):
                    cnname_p1 = os.path.join(output_dir, 'cndata_p1_at{}f.png'.format(i))
                    cnname_p2 = os.path.join(output_dir, 'cndata_p2_at{}f.png'.format(i))
                    cnr.set_img(frame)
                    cnr.write((cnname_p1, cnname_p2))

                if(write_mode == 0 or write_mode == 1):
                    dmname_p1 = os.path.join(output_dir, 'dmdata_p1_at{}f.png'.format(i))
                    dmname_p2 = os.path.join(output_dir, 'dmdata_p2_at{}f.png'.format(i))
                    dr.set_img(frame)
                    dr.write((dmname_p1, dmname_p2))

video.release()
cv2.destroyAllWindows()
