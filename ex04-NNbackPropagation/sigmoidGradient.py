# -*- coding: utf-8 -*-
import numpy as np
from sigmoid import sigmoid


def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))