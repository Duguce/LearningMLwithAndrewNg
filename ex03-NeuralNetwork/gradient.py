import numpy as np
from sigmoid import sigmoid


def gradient(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    error = sigmoid(X * theta.T) - y
    grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)
    grad[0, 0] = np.sum(np.multiply(error, X[:, 0])) / len(X)
    return np.array(grad).ravel()
