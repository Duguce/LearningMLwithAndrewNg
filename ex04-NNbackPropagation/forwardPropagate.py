# -*- coding: utf-8 -*-
import numpy as np
from sigmoid import sigmoid


def forward_propagate(X, theta1, theta2):
    m = X.shape[0]
    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    z2 = a1.dot(theta1.T)
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    z3 = a2.dot(theta2.T)
    h = sigmoid(z3)
    return a1, z2, a2, z3, h
