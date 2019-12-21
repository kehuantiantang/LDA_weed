# coding=utf-8

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, average_precision_score, \
    classification_report
from sklearn.preprocessing import scale

from feature_extractor import create_lbp_texture_feature, create_color_feature, \
    create_hog_feature


def prepare_data(x):
    lbp = create_lbp_texture_feature(x)
    color = create_color_feature(x)
    hog = create_hog_feature(x)

    x_feature = np.concatenate([lbp, color, hog], axis=-1)
    print(x_feature.shape, y_train.shape)
    x_feature = scale(x_feature)

    return x_feature


if __name__ == '__main__':
    # data = np.load("data224.npz")
    # x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data[
    #     'x_test'], data['y_test']
    # print('Preparing the data finish!')
    #
    # X_train_feature = prepare_data(x_train)
    # X_test_feature = prepare_data(x_test)

    data = np.load('feature.npz')
    X_train_feature, X_test_feature, y_train, y_test = data['x_train'], data[
        'x_test'], data['y_train'], data['y_test']

    print('Start training')
    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train_feature, y_train)
    y_pred = clf.predict(X_train_feature)
    print('Training accuracy:', accuracy_score(y_true=y_train, y_pred=y_pred))
    print('Average Training accuracy:',
          average_precision_score(y_true=y_train, y_score=y_pred))
    print(classification_report(y_train, y_pred, target_names=['crop', 'weed']))

    print('Start predicting')
    y_pred = clf.predict(X_test_feature)
    print('Test accuracy:', accuracy_score(y_true=y_test, y_pred=y_pred))
    print('Average Test accuracy:',
          average_precision_score(y_true=y_test, y_score=y_pred))
    print(classification_report(y_test, y_pred, target_names=['crop', 'weed']))

    # np.savez('feature.npz', x_train=X_train_feature, y_train=y_train,  #          x_test=X_test_feature, y_test=y_test)
