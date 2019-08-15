import cv2
import numpy as np
import os
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#########################
# SETTING
#########################
train_data_dir = 'E:/develop/test/trainingdata'             # training data directory
output_pickle = 'testmodel.pickle'                          # output pickle file name
#########################

X = []
y = []

dir_list = os.listdir(train_data_dir)

print('start making training data :)')
for dir in dir_list:
    file_list_path = os.path.join(train_data_dir, dir)
    file_list = os.listdir(file_list_path)
    root, ext = os.path.splitext(dir)
    for file in file_list:
        root, ext = os.path.splitext(file)
        if(ext == '.png'):
            abs_name = os.path.join(file_list_path, file)
            image = cv2.imread(abs_name, 0)
            X.append(image.reshape(-1,))
            y.append(int(dir))

X = np.array(X)
y = np.array(y)
print('complete making training data :D')

print('start training :)')
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = SVC(kernel='linear', random_state=None)
model.fit(X_train, y_train)
print('complete training :)')
pred_train = model.predict(X_train)
accuracy_train = accuracy_score(y_train, pred_train)
print('accuracy at training data set： %.2f' % accuracy_train)

pred_test = model.predict(X_test)
accuracy_test = accuracy_score(y_test, pred_test)
print('accuracy at test data set： %.2f' % accuracy_train)

filepath = os.path.split(__file__)[0]
uri_pickle = os.path.join(filepath, output_pickle)

with open(uri_pickle, mode='wb') as fp:
    pickle.dump(model, fp)
