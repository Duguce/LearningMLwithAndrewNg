# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def plot_data(X, y):
    plt.figure()
    positive = X[y == 1]
    negative = X[y == 0]
    plt.scatter(positive[:, 0], positive[:, 1], marker='+', label='y=1')
    plt.scatter(negative[:, 0], negative[:, 1], marker='o', label='y=0')
    plt.legend()