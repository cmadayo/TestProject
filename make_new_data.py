import cv2
import os
import recognition
import pickle

#########################
# SETTING
#########################
sample_data_dir = 'E:/develop/test/sampledata'             # training data directory
picklefile = 'model3.pickle'                               # pickle file of model data
output_dir = 'E:/develop/test/trainingdata'                # output directory
progress_step = 1000                                       # step count for debug log
#########################

file_list = os.listdir(sample_data_dir)
dr = recognition.DamageRecogition()

filepath = os.path.split(__file__)[0]
uri_pickle = os.path.join(filepath, picklefile)

with open(uri_pickle, mode='rb') as fp:
    model = pickle.load(fp)

for i, file_name in enumerate(file_list):
    if(i % progress_step == 0):
        print('Read progress rate {}/{}'.format(i, len(file_list)))

    abs_name = os.path.join(sample_data_dir, file_name)
    image_org = cv2.imread(abs_name, 0)

    number = str(model.predict(image_org.reshape(1, -1))[0])

    output_sub_dir = os.path.join(output_dir, number)
    output_filename = os.path.join(output_sub_dir, '{}.png'.format(i))

    cv2.imwrite(output_filename, image_org)
