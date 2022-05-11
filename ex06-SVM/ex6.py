# -*- coding: utf-8 -*-
import numpy as np
from scipy.io import loadmat
from sklearn import svm
import itertools
from plotData import plot_data


def main():
    print('Loading and Visualizing ex6data1.mat ...')
    data = loadmat('data/ex6data1.mat')
    X = data['X']
    y = data['y'].flatten()
    plot_data(X, y)

    print('Training Linear SVM ...')
    svc = svm.SVC(C=100, kernel='linear', tol=1e-3)
    svc.fit(X, y)
    print('Training accuracy = {0}%'.format(np.round(svc.score(X, y) * 100, 2)))

    print('Loading and Visualizing ex6data2.mat ...')
    data = loadmat('data/ex6data2.mat')
    X = data['X']
    y = data['y'].flatten()
    plot_data(X, y)

    print('Training SVM ...')
    svc2 = svm.SVC(C=100, gamma=10, probability=True)
    svc2.fit(X, y)
    print('Training accuracy = {0}%'.format(np.round(svc2.score(X, y) * 100, 2)))

    print('Loading and Visualizing ex6data3.mat ...')
    data = loadmat('data/ex6data3.mat')
    X = data['X']
    y = data['y'].flatten()
    plot_data(X, y)

    print('Training SVM ...')
    svc3 = svm.SVC(C=100, kernel='rbf', gamma=np.power(0.1, -2))
    svc3.fit(X, y)
    print('Training accuracy = {0}%'.format(np.round(svc3.score(X, y) * 100, 2)))

    print('Loading spam ...')
    spam_train = loadmat('data/spamTrain.mat')
    spam_test = loadmat('data/spamTest.mat')
    X = spam_train['X']
    Xtest = spam_test['Xtest']
    y = spam_train['y'].ravel()
    ytest = spam_test['ytest'].ravel()
    svc4 = svm.SVC()
    svc4.fit(X, y)
    print('Training accuracy = {0}%'.format(np.round(svc4.score(X, y) * 100, 2)))
    print('Test accuracy = {0}%'.format(np.round(svc4.score(Xtest, ytest) * 100, 2)))


if __name__ == '__main__':
    main()
