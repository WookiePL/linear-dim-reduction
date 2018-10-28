import time
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import train_test_split


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
    time.sleep(1)


def standardise_classes(y):
    for index, item in enumerate(y):
        if not (item == 0):
            y[index] = 1
    return y


def plot_decision_regions(X, y, classifier, name, resolution=0.02, **kwargs):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v', '+')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan', 'orange')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    # calculate and print prediction accuracy
    pred_test = classifier.predict(X)
    method = kwargs.get('method', 'PCA')
    print('\nPrediction accuracy for the %s dataset with %s' % (name, method))
    print('{:.2%}\n'.format(metrics.accuracy_score(y, pred_test)))

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


# TO REMOVE
def show_prediction_accuracy(y_test, pred_test, pred_test_std):
    # Show prediction accuracies in scaled and unscaled data.
    print('\nPrediction accuracy for the normal test dataset with PCA')
    print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test)))

    print('\nPrediction accuracy for the standardized test dataset with PCA')
    print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test_std)))


# TO REMOVE
def plot_colors():
    plt.plot([1, 2], lw=4, c='tab:green')
    plt.show()


# PROBABLY TO REMOVE
def preprocess_heart_disease_data(url):
    # załadowanie zbioru danych do Pandas DataFrame
    df = pd.read_csv(url,
                     names=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
                            'slope', 'ca', 'thal', 'num, the predicted attribute'])

    df = df.replace('?', '0')

    df.convert_objects(convert_numeric=True)

    # podział na zbiór cech i klasy
    X, y = df.iloc[:, :13].values, df.iloc[:, 13].values

    # y = standardise_classes(y)

    # podział danych na 70% zbiór treningowy, 30% testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    return X, y, X_train, X_test, y_train, y_test


def ignore_all_warnings():
    warnings.simplefilter(action='ignore', category=DataConversionWarning)