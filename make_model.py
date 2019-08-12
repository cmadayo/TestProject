import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import recogition
import pyocr
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import pickle

def shuffle(matrix, target, test_proportion):
    ratio = int(matrix.shape[0]/test_proportion) #should be int
    X_train = matrix[ratio:,:]
    X_test =  matrix[:ratio,:]
    Y_train = target[ratio:,:]
    Y_test =  target[:ratio,:]
    return X_train, X_test, Y_train, Y_test


zero_arr = np.array([0 for i in range(0, 226)])
one_arr = np.array([1 for i in range(226, 941)])
two_arr = np.array([2 for i in range(941, 1325)])
three_arr = np.array([3 for i in range(1325, 1566)])
four_arr = np.array([4 for i in range(1566, 1924)])
five_arr = np.array([5 for i in range(1924, 2612)])
six_arr = np.array([6 for i in range(2612, 2906)])
seven_arr = np.array([7 for i in range(2906, 3126)])
eight_arr = np.array([8 for i in range(3126, 3520)])
nine_arr = np.array([9 for i in range(3520, 3817)])
other_arr = np.array([10 for i in range(3817, 4201)])

X = []
y = np.concatenate([zero_arr, one_arr, two_arr, three_arr, four_arr, five_arr, six_arr, seven_arr, eight_arr, nine_arr, other_arr])

data_dir_path = 'E:/data5/all'
file_list = os.listdir('E:/data5/all')

for file_name in file_list:
    root, ext = os.path.splitext(file_name)
    abs_name = os.path.join(data_dir_path, file_name)
    image = cv2.imread(abs_name, 0)
    orgHeight, orgWidth = image.shape[:2]
    size = (int(orgWidth/4), int(orgHeight/4))
    image = cv2.resize(image, size)
    # print(image.reshape(-1,).shape)
    X.append(image.reshape(-1,))

X = np.array(X)
y = np.array(y)

# print(X.shape)
# print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y)
print(X_test[0].shape)
#alpha_val = 0.0001
#mlpc = MLPClassifier(hidden_layer_sizes=(26, ), solver="adam", random_state=9999, max_iter=1000, alpha=alpha_val)
#mlpc.fit(X_train, y_train)
#acc = mlpc.score(X_test, y_test)*100

lr = LogisticRegression()
lr.fit(X_train, y_train)
acc = lr.score(X_test, y_test)*100

print('acc={}'.format(acc))

# fileName = 'model.pickle'
# filepath = os.path.split(__file__)[0]
# uri_pickle = os.path.join(filepath, fileName)
#
# with open(uri_pickle, mode='wb') as fp:
#     pickle.dump(lr, fp)

 #    txt = tool.image_to_string(
 #        Image.fromarray(image),
 #        builder=pyocr.builders.DigitBuilder(tesseract_layout=6)
 #    )
 #    output = os.path.join(data2_dir_path, '{}_{}.png'.format(txt, i))
 #    cv2.imwrite(output, image, [cv2.IMWRITE_JPEG_QUALITY, 100])

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
