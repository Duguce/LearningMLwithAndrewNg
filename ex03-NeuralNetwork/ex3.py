# -*- coding: utf-8 -*-
from scipy.io import loadmat
import numpy as np
from oneVsAll import one_vs_all
from predictOneVsAll import predict_one_vs_all


def main():
    # 导入数据集
    data = loadmat('ex3data1.mat')

    rows = data['X'].shape[0]

    y_0 = np.array([1 if label == 0 else 0 for label in data['y']])
    y_0 = np.reshape(y_0, (rows, 1))

    all_theta = one_vs_all(data['X'], data['y'], 10, 1)

    y_pred = predict_one_vs_all(data['X'], all_theta)
    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, data['y'])]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print('Training set accuracy = {0}%'.format(accuracy * 100))


if __name__ == '__main__':
    main()
