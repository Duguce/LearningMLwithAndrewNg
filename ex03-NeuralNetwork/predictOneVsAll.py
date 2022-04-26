import numpy as np
from sigmoid import sigmoid


def predict_one_vs_all(X, all_theta):
    X = np.column_stack((np.ones(X.shape[0]), X))
    X = np.matrix(X)
    all_theta = np.matrix(all_theta)
    h = sigmoid(X * all_theta.T)
    h_argmax = np.argmax(h, axis=1)
    h_argmax = h_argmax + 1
    return h_argmax
