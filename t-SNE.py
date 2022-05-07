
# coding='utf-8'
"""t-SNE对手写数字进行可视化"""
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE

import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets


def print_rawdata():
    digits = datasets.load_digits(n_class=10)
    X, y = digits.data, digits.target
    n_samples, n_features = X.shape

    '''显示原始数据'''
    n = 10  # 10*10 pictures
    img = np.zeros((10 * n, 10 * n))
    for i in range(n):
        ix = 10 * i + 1
        for j in range(n):
            iy = 10 * j + 1
            img[ix:ix + 8, iy:iy + 8] = X[i * n + j].reshape((8, 8))
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def get_data():
    digits = datasets.load_digits(n_class=10)
    data = digits.data
    label = digits.target
    n_samples, n_features = data.shape
    return data, label, n_samples, n_features


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def main():
    print_rawdata()
    data, label, n_samples, n_features = get_data()
    print('Computing t-SNE embedding',n_samples, n_features)
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time.time()
    result = tsne.fit_transform(data)
    fig = plot_embedding(result, label,
                            't-SNE embedding of the digits (time %.2fs)'
                            % (time.time() - t0))
    plt.show()


if __name__ == '__main__':
    main()
