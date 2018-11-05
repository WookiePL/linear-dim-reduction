import os

from reduction.classifier_factory import ClassifierFactory
from reduction.utils import plot_decision_regions, save_plot_as_png_file
from reduction_dermatology.derm_utils import preprocess_dermatology_data
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from reduction_dermatology.results_metrics import count_print_confusion_matrix
from report_model.input_params import InputParams


def process_lda(url, title, n_components, **kwargs):
    input_params = InputParams(os.path.basename(__file__), url, title, n_components, kwargs.get('classifier', 'lr'))

    method_name = 'LDA'

    X, y, X_train, X_test, y_train, y_test = preprocess_dermatology_data(url)


    # standaryzacja danych
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    np.set_printoptions(precision=4) #ilosc miejsc po przecinku dla liczb w macierzach
    mean_vectors = []
    for label in range(1, 7):
        mean_vectors.append(np.mean(X_train_std[y_train == label], axis=0))
        print('MeanVector %s: %s\n' % (label, mean_vectors[label - 1]))

    # liczba cech
    d = 34
    S_W = np.zeros((d, d))
    for label, mv in zip(range(1, 7), mean_vectors):
        class_scatter = np.zeros((d, d))
        for row in X_train_std[y_train == label]:
            row, mv = row.reshape(d, 1), mv.reshape(d, 1)
            class_scatter += (row - mv).dot((row - mv).T)
            S_W += class_scatter
    print('within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))

    y_train.dtype
    y_train = y_train.astype(np.int32)

    print('Class label distribution: %s'
          % np.bincount(y_train)[1:])



    d = 34  # number of features
    S_W = np.zeros((d, d))
    for label, mv in zip(range(1, 7), mean_vectors):
        class_scatter = np.cov(X_train_std[y_train == label].T)
        S_W += class_scatter
    print('Scaled within-class scatter matrix: %sx%s' % (S_W.shape[0],
                                                         S_W.shape[1]))


    mean_overall = np.mean(X_train_std, axis=0)
    d = 34  # number of features
    S_B = np.zeros((d, d))
    for i, mean_vec in enumerate(mean_vectors):
        n = X_train[y_train == i + 1, :].shape[0]
        mean_vec = mean_vec.reshape(d, 1)  # make column vector
        mean_overall = mean_overall.reshape(d, 1)  # make column vector
        S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

    print('Between-class scatter matrix: %sx%s' % (S_B.shape[0], S_B.shape[1]))



    eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

    # Make a list of (eigenvalue, eigenvector) tuples
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
                   for i in range(len(eigen_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

    # Visually confirm that the list is correctly sorted by decreasing eigenvalues
    print('Eigenvalues in decreasing order:\n')
    for eigen_val in eigen_pairs:
        print(eigen_val[0])


    tot = sum(eigen_vals.real)
    discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
    cum_discr = np.cumsum(discr)

    plt.bar(range(1, 35), discr, alpha=0.5, align='center',
            label='pojedyncza rozróżnialność')
    plt.step(range(1, 35), cum_discr, where='mid',
             label='łączna rozróżnialność')
    plt.ylabel('współczynnik rozróżnialności')
    plt.xlabel('liniowe dyskryminanty')
    plt.ylim([-0.1, 1.1])
    plt.legend(loc='best')
    plt.tight_layout()
    # plt.savefig('./figures/lda1.png', dpi=300)
    plt.show()

    w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
                   eigen_pairs[1][1][:, np.newaxis].real))
    print('Matrix W:\n', w)




    X_train_lda = X_train_std.dot(w)
    # colors = ['r', 'b', 'g']
    # markers = ['s', 'x', 'o']
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')

    for l, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(X_train_lda[y_train == l, 0] * (-1),
                    X_train_lda[y_train == l, 1] * (-1),
                    c=c, label=l, marker=m)

    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend(loc='lower right')
    plt.tight_layout()
    # plt.savefig('./figures/lda2.png', dpi=300)
    plt.show()


    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

    lda = LDA(n_components=n_components)
    X_train_lda = lda.fit_transform(X_train_std, y_train)
    X_test_lda = lda.transform(X_test_std)

    # from sklearn.linear_model import LogisticRegression
    # lr = LogisticRegression()
    # lr = lr.fit(X_train_lda, y_train)
    _classifier = ClassifierFactory.get_classifier(kwargs)
    _classifier.fit(X_train_lda, y_train)

    training_png_url = ''
    test_png_url = ''

    if n_components == 2:
        plot_decision_regions(X_train_lda, y_train, classifier=_classifier, name="%s training" % title, method=method_name)
        plt.xlabel('LD 1')
        plt.ylabel('LD 2')
        plt.title(title + ', 2 component LDA, zbiór treningowy')
        plt.legend(loc='lower left')
        plt.tight_layout()
        training_png_url = save_plot_as_png_file(plt)
        plt.show()

        plot_decision_regions(X_test_lda, y_test, classifier=_classifier, name="%s test" % title, method=method_name)
        plt.xlabel('LD 1')
        plt.ylabel('LD 2')
        plt.title(title + ', 2 component LDA, zbiór testowy')
        plt.legend(loc='lower left')
        plt.tight_layout()
        test_png_url = save_plot_as_png_file(plt)
        plt.show()

    count_print_confusion_matrix(X_train_lda, X_test_lda, y_train, y_test, _classifier,
                                 run_id=kwargs.get('run_id', '0'),
                                 input_params=input_params,
                                 training_png_url=training_png_url,
                                 test_png_url=test_png_url)
    pass


url1 = "D:\\mgr\\dermatology\\dermatology.data"

process_lda(url1, 'Dermatology', n_components=2)
