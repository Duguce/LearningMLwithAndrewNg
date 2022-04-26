# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def computeCost(X, y, theta):
    """代价函数"""
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


def gradientDescent(X, y, theta, alpha=0.01, num_iters=1500):
    """批量梯度下降算法"""
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    for i in range(iters):
        error = (X * theta.T) - y
        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X))) * np.sum(term)
        theta = temp
        cost[i] = computeCost(X, y, theta)
    return theta, cost


if __name__ == '__main__':
    data_path = 'ex1data2.txt'
    data = pd.read_csv(data_path, header=None, names=['Size', 'Bedrooms', 'Price'])
    data = (data - np.mean(data)) / np.std(data)
    alpha = 0.01
    iters = 1500
    data.insert(0, 'Ones', 1)
    cols = data.shape[1]
    X = data.iloc[:, :cols - 1]
    y = data.iloc[:, cols - 1:cols]
    X = np.matrix(X.values)
    y = np.matrix(y.values)
    theta = np.matrix(np.array([0, 0, 0]))
    g, cost = gradientDescent(X, y, theta, alpha, iters)
    computeCost(X, y, g)
    fig, ax = plt.subplots()
    ax.plot(np.arange(iters), cost, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')
    plt.show()
