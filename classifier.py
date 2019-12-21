# coding=utf-8

import numpy as np
from sklearn import linear_model, clone, metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, average_precision_score, \
    classification_report
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale

from feature_extractor import create_lbp_texture_feature, create_color_feature, \
    create_hog_feature


def prepare_data(x):
    lbp = create_lbp_texture_feature(x)
    color = create_color_feature(x)
    hog = create_hog_feature(x)

    x_feature = np.concatenate([lbp, color, hog], axis=-1)
    x_feature = scale(x_feature)

    return x_feature

def train_test_data(is_feature = True):
    # data = np.load("data224.npz")
    # x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data[
    #     'x_test'], data['y_test']
    # print('Preparing the data finish!')
    #
    # X_train_feature = prepare_data(x_train)
    # X_test_feature = prepare_data(x_test)

    if is_feature:
        data = np.load('feature.npz')
        X_train_feature, X_test_feature, y_train, y_test = data['x_train'], data[
            'x_test'], data['y_train'], data['y_test']
        return X_train_feature, y_train, X_test_feature, y_test
    else:
        data = np.load("data224.npz")
        return data['x_train'], data['y_train'], data[
            'x_test'], data['y_test']

def lda():
    X_train_feature, X_test_feature, y_train, y_test = train_test_data()

    print('Start training')
    # clf = LinearDiscriminantAnalysis()
    clf = QuadraticDiscriminantAnalysis()
    clf.fit(X_train_feature, y_train)
    y_pred = clf.predict(X_train_feature)
    result_analysis(y_pred, y_train)

    print('Start predicting')
    y_pred = clf.predict(X_test_feature)
    result_analysis(y_pred, y_test)

    # np.savez('feature.npz', x_train=X_train_feature, y_train=y_train,  #          x_test=X_test_feature, y_test=y_test)

def result_analysis(y_pred, y_truth):
    print('Test accuracy:', accuracy_score(y_true=y_truth, y_pred=y_pred))
    print('Average Test accuracy:',
          average_precision_score(y_true=y_truth, y_score=y_pred))
    print(classification_report(y_truth, y_pred, target_names=['crop', 'weed']))

def rbm():
    X_train, Y_train, X_test, Y_test = train_test_data(
        is_feature = False)

    rbm = BernoulliRBM(random_state=0, verbose=True)
    logistic = linear_model.LogisticRegression(solver='newton-cg', tol=1)
    rbm_features_classifier = Pipeline(
        steps=[('rbm', rbm), ('logistic', logistic)])

    rbm.learning_rate = 0.06
    rbm.n_iter = 10
    # More components tend to give better prediction performance, but larger
    # fitting time
    rbm.n_components = 100
    logistic.C = 6000

    # Training RBM-Logistic Pipeline
    rbm_features_classifier.fit(X_train, Y_train)

    # Training the Logistic regression classifier directly on the pixel
    raw_pixel_classifier = clone(logistic)
    raw_pixel_classifier.C = 100.
    raw_pixel_classifier.fit(X_train, Y_train)

    Y_pred = rbm_features_classifier.predict(X_test)
    print("Logistic regression using RBM features:\n%s\n" % (
        metrics.classification_report(Y_test, Y_pred)))

    Y_pred = raw_pixel_classifier.predict(X_test)

if __name__ == '__main__':
    pass
