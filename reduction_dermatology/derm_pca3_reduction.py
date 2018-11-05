import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from reduction.classifier_factory import ClassifierFactory
from reduction.utils import save_plot_as_png_file, standardise_classes, plot_decision_regions, plot_colors
from reduction_dermatology.derm_utils import ignore_all_warnings
from reduction_dermatology.results_metrics import count_print_confusion_matrix
from report_model.input_params import InputParams


def process_pca(url, title, n_components, **kwargs):
    input_params = InputParams(os.path.basename(__file__), url, title, n_components, kwargs.get('classifier', 'lr'))

    ignore_all_warnings()
    print('{}, {} component PCA'.format(title, n_components))
    # załadowanie zbioru danych do Pandas DataFrame
    df = pd.read_csv(url,
                     names=['erythema',
                            'scaling',
                            'definite borders',
                            'itching',
                            'koebner phenomenon',
                            'polygonal papules',
                            'follicular papules',
                            'oral mucosal involvement',
                            'knee and elbow involvement',
                            'scalp involvement',
                            'family history, (0 or 1)',
                            'melanin incontinence',
                            'eosinophils in the infiltrate',
                            'PNL infiltrate',
                            'fibrosis of the papillary dermis',
                            'exocytosis',
                            'acanthosis',
                            'hyperkeratosis',
                            'parakeratosis',
                            'clubbing of the rete ridges',
                            'elongation of the rete ridges',
                            'thinning of the suprapapillary epidermis',
                            'spongiform pustule',
                            'munro microabcess',
                            'focal hypergranulosis',
                            'disappearance of the granular layer',
                            'vacuolisation and damage of basal layer',
                            'spongiosis',
                            'saw-tooth appearance of retes',
                            'follicular horn plug',
                            'perifollicular parakeratosis',
                            'inflammatory monoluclear inflitrate',
                            'band-like infiltrate',
                            'Age (linear)',
                            'class'])

    df = df.replace('?', '0')
    df.convert_objects(convert_numeric=True)

    # podział na zbiór cech i klasy
    X, y = df.iloc[:, :34].values, df.iloc[:, 34].values


    # podział danych na 70% zbiór treningowy, 30% testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # standaryzacja danych
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    # otrzymywanie wartości własnych (eigenvalues)
    cov_mat = np.cov(X_train_std.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    print('Eigenvalues \n%s' % eigen_vals)

    # suma wariancji
    total = sum(eigen_vals)
    # explained variances
    var_exp = [(i / total) for i in sorted(eigen_vals, reverse=True)]
    # cumulative sum of explained variances
    cum_var_exp = np.cumsum(var_exp)

    plt.bar(range(1, 35), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
    plt.step(range(1, 35), cum_var_exp, where='mid',
             label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # transformacja cech

    # Make a list of (eigenvalue, eigenvector) tuples
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
                   for i in range(len(eigen_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eigen_pairs.sort(key=lambda k: k[0], reverse=True)

    w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
                   eigen_pairs[1][1][:, np.newaxis]))
    print('Matrix W:\n', w)
    X_train_pca = X_train_std.dot(w)
    markers = ('s', 'x', 'o', '^', 'v', '+')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan', 'orange')

    for l, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(X_train_pca[y_train == l, 0],
                    X_train_pca[y_train == l, 1],
                    c=c, label=l, marker=m)

    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.tight_layout()

    plt.show()

   # X_train_std[0].dot(w)

    pca = PCA()
    X_train_pca = pca.fit_transform(X_train_std)
    pca.explained_variance_ratio_
    plt.bar(range(1, 35), pca.explained_variance_ratio_, alpha=0.5, align='center')
    plt.step(range(1, 35), np.cumsum(pca.explained_variance_ratio_), where='mid')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.show()

    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)



    # lr = LogisticRegression()
    # lr = lr.fit(X_train_pca, y_train)
    #
    # #TODO implement different learning methods
    # svm = SVC(kernel='rbf', random_state=0, gamma=0.7, C=1.0)
    # svm.fit(X_train_pca, y_train)
    _classifier = ClassifierFactory.get_classifier(kwargs)
    _classifier.fit(X_train_pca, y_train)

    training_png_url = ''
    test_png_url = ''

    if n_components == 2:
        plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1])
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.show()

        plot_decision_regions(X_train_pca, y_train, classifier=_classifier, name="%s training" % title)
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.title(title + ', 2 component PCA, zbiór treningowy')
        plt.legend(loc='upper left')
        plt.tight_layout()
        training_png_url = save_plot_as_png_file(plt)
        plt.show()

        plot_decision_regions(X_test_pca, y_test, classifier=_classifier, name="%s test" % title)
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.title(title + ', 2 component PCA, zbiór testowy')
        plt.legend(loc='upper left')
        plt.tight_layout()
        test_png_url = save_plot_as_png_file(plt)
        plt.show()

    count_print_confusion_matrix(X_train_pca,
                                 X_test_pca,
                                 y_train,
                                 y_test,
                                 _classifier,
                                 run_id=kwargs.get('run_id', '0'),
                                 input_params=input_params,
                                 training_png_url=training_png_url,
                                 test_png_url=test_png_url)
    pass


url1 = "D:\\mgr\\dermatology\\dermatology.data"

process_pca(url1, 'Dermatology', 2, classifier='forest')
