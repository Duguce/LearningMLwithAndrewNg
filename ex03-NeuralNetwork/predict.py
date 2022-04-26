import numpy as np
from sigmoid import sigmoid


def predictNN(theta1, theta2, a1):
    z = sigmoid(a1.dot(theta1.T))
    a2 = np.column_stack((np.ones(z.shape[0]), z))
    a3 = sigmoid(a2.dot(theta2.T))
    p = np.argmax(a3, axis=1)
    p = p + 1
    return p
