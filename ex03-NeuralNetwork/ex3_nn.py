# -*- coding: utf-8 -*-
import scipy.io as scio
import numpy as np
from oneVsAll import one_vs_all
from predict import predictNN
from displayData import display_data
import pandas as pd


def main():
    input_layer_size = 400  # 输入层的单元数  原始输入特征数 20*20=400
    hidden_layer_size = 25  # 隐藏层 25个神经元
    num_labels = 10  # 10个标签 数字0对应类别10  数字1-9对应类别1-9
    print('Loading and Visualizing Data ...')
    data = scio.loadmat('ex3data1.mat')  # 读取数据
    X = data['X']  # 获取输入特征矩阵 5000*400
    y = data['y'].flatten()  # 获取5000个样本的标签 用flatten()函数 将5000*1的2维数组 转换成包含5000个元素的一维数组
    m = y.size  # 样本数 5000
    # 随机选100个样本 可视化
    rand_indices = np.random.permutation(range(m))
    selected = X[rand_indices[:100], :]
    display_data(selected)
    print('Loading Saved Neural Network Parameters ...')
    X = np.column_stack((np.ones(X.shape[0]), X))
    data = scio.loadmat('ex3weights.mat')  # 读取参数数据
    theta1 = data['Theta1']  # 输入层和隐藏层之间的参数矩阵
    theta2 = data['Theta2']  # 隐藏层和输出层之间的参数矩阵
    print(theta1.shape)
    print(theta2.shape)
    pred = predictNN(theta1, theta2, X)
    print('Training set accuracy: {}%'.format(np.mean(pred == y) * 100))


if __name__ == '__main__':
    main()
