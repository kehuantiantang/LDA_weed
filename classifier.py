# coding=utf-8

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, average_precision_score, \
    classification_report
from sklearn.preprocessing import scale

from feature_extractor import create_lbp_texture_feature, create_color_feature, \
    create_hog_feature

data = np.load("data224.npz")
x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data[
    'x_test'], data['y_test']
print('Preparing the data finish!')

train_lbp = create_lbp_texture_feature(x_train)
train_color = create_color_feature(x_train)
train_hog = create_hog_feature(x_train)

X_train_feature = np.concatenate([train_lbp, train_color, train_hog], axis=-1)
print(X_train_feature.shape, y_train.shape)
X_train_feature = scale(X_train_feature)

print('Start training')
clf = LinearDiscriminantAnalysis()
clf.fit(X_train_feature, y_train)

print('Start predicting')
test_lbp = create_lbp_texture_feature(x_test)
test_color = create_color_feature(x_test)
test_hog = create_hog_feature(x_test)
X_test_feature = np.concatenate([test_lbp, test_color, test_hog], axis=-1)
print(X_test_feature.shape, y_test.shape)
X_test_feature = scale(X_test_feature)
y_pred = clf.predict(X_test_feature)
print('Test accuracy:', accuracy_score(y_true=y_test, y_pred=y_pred))
print('Average Test accuracy:',
      average_precision_score(y_true=y_test, y_score=y_pred))
print(classification_report(y_test, y_pred, target_names=['crop', 'weed']))

np.savez('feature.npz', x_train = X_train_feature, y_train = y_train, x_test
= X_test_feature, y_test = y_test)

