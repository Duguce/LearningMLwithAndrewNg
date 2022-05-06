# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.io import loadmat
from displayData import display_data
from sklearn.preprocessing import OneHotEncoder
from backpropReg import backpropReg
from scipy.optimize import minimize
from forwardPropagate import forward_propagate


def main():
    data = loadmat('ex4data1.mat')
    print('[NORMAL] 成功地加载数据集...')
    X = data['X']
    y = data['y']
    print(f'X的大小为{X.shape}, y的大小为{y.shape}')
    # 对y标签进行一次one-hot编码
    encoder = OneHotEncoder(sparse=False)
    y_onehot = encoder.fit_transform(y)
    m = X.shape[0]
    rand_indices = np.random.permutation(range(m))
    selected = X[rand_indices[:100], :]
    fig = display_data(selected)
    fig.savefig('Example01')
    print('[NORMAL] 成功地从原始数据集随机选100个样本进行可视化...')
    # 初始化设置
    input_size = 400
    hidden_size = 25
    num_labels = 10
    learning_rate = 1
    # 随机初始化完整网络参数大小的参数数组
    params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25
    X = np.matrix(X)
    y = np.matrix(y)

    J, grad = backpropReg(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)
    print('[NORMAL] 对网络中的参数进行训练...')
    fmin = minimize(fun=backpropReg, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate),
                    method='TNC', jac=True, options={'maxiter': 250})
    print('[NORMAL] 训练成功...')
    X = np.matrix(X)
    theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    y_pred = np.array(np.argmax(h, axis=1) + 1)
    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print('accuracy = {0}%'.format(accuracy * 100))


if __name__ == '__main__':
    main()
