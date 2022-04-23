# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt


def loadData():
    """载入数据集"""
    data_path = 'ex2data2.txt'
    data = pd.read_csv(data_path, header=None, names=['Test 1', 'Test 2', 'Accepted'])
    return data


def plotData():
    """将数据集进行可视化"""
    data = loadData()
    # 区分pos和neg的数据项
    pos = data[data['Accepted'] == 1]
    neg = data[data['Accepted'] == 0]
    # 可视化展示
    fig, ax = plt.subplots()
    ax.scatter(pos['Test 1'], pos['Test 2'], s=50,
               c='y', marker='o', label='Accepted')
    ax.scatter(neg['Test 1'], neg['Test 2'], s=50,
               c='r', marker='*', label='Rejected')
    ax.legend()
    ax.set_xlabel('Test 1 Score')
    ax.set_ylabel('Test 2 Score')
    plt.show()


def sigmoid(z):
    """定义sigmoid函数"""
    return 1 / (1 + np.exp(-z))


def predict(theta, X):
    """计算预测值"""
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


def costReg(theta, X, y, learningRate):
    """正则化代价函数"""
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply(1 - y, np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    return np.sum(first - second) / len(X) + reg


def gradientReg(theta, X, y, learningRate):
    """计算正则化梯度步长"""
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmoid(X * theta.T) - y
    for i in range(parameters):
        reg = (learningRate / len(X)) * theta[:, i]
        term = np.multiply(error, X[:, i])
        grad[i] = np.sum(term) / len(X) if (i == 0) else np.sum(term) / len(X) + reg
    return grad


def calAccuracy():
    """计算accuracy值"""
    # 参数寻优
    result = opt.fmin_tnc(func=costReg, x0=theta, fprime=gradientReg, args=(X, y, learningRate))

    theta_min = np.matrix(result[0])
    predictions = predict(theta_min, X)
    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
    accuracy = (sum(map(int, correct)) / len(correct))
    print('accuracy = {0}'.format(accuracy))


def mapFeature(data, degree):
    """正则化"""
    x1 = data['Test 1']
    x2 = data['Test 2']
    data.insert(3, 'Ones', 1)
    for i in range(1, degree):
        for j in range(i):
            data[f'F{str(i)}{str(j)}'] = np.power(x1, i - j) * np.power(x2, j)
    data.drop('Test 1', axis=1, inplace=True)
    data.drop('Test 2', axis=1, inplace=True)
    mapData = data
    return mapData


if __name__ == '__main__':
    plotData()
    # 创建一组多项式特征
    data = loadData()
    degree = 5
    mapData = mapFeature(data, degree)
    # 初始化变量
    cols = mapData.shape[1]
    X = mapData.iloc[:, 1:cols]
    y = mapData.iloc[:, :1]
    X = np.array(X.values)
    y = np.array(y.values)
    theta = np.zeros(11)
    learningRate = 1
    # 计算accuracy
    calAccuracy()
