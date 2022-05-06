# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


def display_data(data):
    """数据集可视化展示"""
    (m, n) = data.shape
    example_width = np.round(np.sqrt(n)).astype(int)
    example_height = (n / example_width).astype(int)
    display_rows = np.floor(np.sqrt(m)).astype(int)
    display_cols = np.ceil(m / display_rows).astype(int)
    pad = 1
    display_array = - np.ones((pad + display_rows * (example_height + pad),
                               pad + display_rows * (example_height + pad)))
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex > m:
                break
            max_val = np.max(np.abs(data[curr_ex]))
            display_array[pad + j * (example_height + pad) + np.arange(example_height),
                          pad + i * (example_width + pad) + np.arange(example_width)[:, np.newaxis]] = \
                data[curr_ex].reshape((example_height, example_width)) / max_val
            curr_ex += 1
        if curr_ex > m:
            break
    fig = plt.figure()
    plt.imshow(display_array, cmap='gray', extent=[-1, 1, -1, 1])
    plt.axis('off')
    return fig
