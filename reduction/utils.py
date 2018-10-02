import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def make_num_binary(item):
    if not (item == 0):
        return 1
    else:
        return 0


def save_plot_as_png_file(plt):
    FOLDER_NAME = "plots/"
    plt2 = plt
    date = str(datetime.now().strftime('%Y-%m-%d %H%M%S'))

    plt2.savefig(FOLDER_NAME + date + ".png")
    time.sleep(0.5)


def standardise_classes(y):
    for index, item in enumerate(y):
        if not (item == 0):
            y[index] = 1
    return y


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.6,
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx],
                    label=cl)
