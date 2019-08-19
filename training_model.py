import cv2
import numpy as np
import os
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pathlib import Path

#########################
# SETTING
#########################
train_data_dir = 'E:/develop/test/trainingdata'             # training data directory
output_pickle = 'testmodel.pickle'                          # output pickle file name
#########################

X = []
y = []

print('start reading training data :)')
p_label_dir = Path(train_data_dir)
it_dir = p_label_dir.iterdir()

for dir in it_dir:
    if dir.is_dir():
        p_filedir = Path(dir)
        it_file = p_filedir.glob('*.png')
    else:
        continue

    for file in it_file:
        file = str(file)
        image = cv2.imread(file, 0)
        X.append(image.reshape(-1,))
        y.append(int(dir.name))

X = np.array(X)
y = np.array(y)

print('read complete :D')

print('start training :O')
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = SVC(kernel='linear', random_state=None)
model.fit(X_train, y_train)
print('complete training :P')

print('culculate accuracy score...')
pred_train = model.predict(X_train)
accuracy_train = accuracy_score(y_train, pred_train)
print('accuracy score at training data set： %.2f' % accuracy_train)

print('culculate accuracy score...')
pred_test = model.predict(X_test)
accuracy_test = accuracy_score(y_test, pred_test)
print('accuracy score at test data set： %.2f' % accuracy_train)

filepath = os.path.split(__file__)[0]
uri_pickle = os.path.join(filepath, output_pickle)

with open(uri_pickle, mode='wb') as fp:
    pickle.dump(model, fp)
