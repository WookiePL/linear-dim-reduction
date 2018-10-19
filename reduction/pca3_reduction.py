import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from reduction.results_metrics import count_print_confusion_matrix
from reduction.utils import save_plot_as_png_file, standardise_classes, plot_decision_regions


def process_pca(url, title):
    # załadowanie zbioru danych do Pandas DataFrame
    df = pd.read_csv(url,
                     names=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
                            'slope', 'ca', 'thal', 'num, the predicted attribute'])

    df = df.replace('?', '0')
    df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
                  'ca', 'thal', 'num, the predicted attribute']
    df.convert_objects(convert_numeric=True)
    print(df.dtypes)

    # podział na zbiór cech i klasy
    X, y = df.iloc[:, :13].values, df.iloc[:, 13].values

    #standaryzacja klas (wartosci > 1 => 1)
    y = standardise_classes(y)

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

    plt.bar(range(1, 14), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
    plt.step(range(1, 14), cum_var_exp, where='mid',
             label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # transformacja cech
    # make a list of (eigenvalue, eigenvector) tuples
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
                   for i in range(len(eigen_vals))]

    # sort the (eigenvalue, eigenvector) tuples from high to low
    eigen_pairs.sort(key=lambda k: k[0], reverse=True)

    w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
                   eigen_pairs[1][1][:, np.newaxis]))
    print('Matrix W:\n', w)
    X_train_pca = X_train_std.dot(w)
    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o']

    for l, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(X_train_pca[y_train == l, 0],
                    X_train_pca[y_train == l, 1],
                    c=c, label=l, marker=m)

    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.tight_layout()

    plt.show()

    X_train_std[0].dot(w)

    pca = PCA()
    X_train_pca = pca.fit_transform(X_train_std)
    pca.explained_variance_ratio_
    plt.bar(range(1, 14), pca.explained_variance_ratio_, alpha=0.5, align='center')
    plt.step(range(1, 14), np.cumsum(pca.explained_variance_ratio_), where='mid')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.show()

    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)

    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1])
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.show()

    lr = LogisticRegression()
    lr = lr.fit(X_train_pca, y_train)

    plot_decision_regions(X_train_pca, y_train, classifier=lr, name="%s training" % title)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title(title + ', 2 component PCA, zbiór treningowy')
    plt.legend(loc='lower left')
    plt.tight_layout()
    save_plot_as_png_file(plt)
    plt.show()

    plot_decision_regions(X_test_pca, y_test, classifier=lr, name="%s test" % title)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title(title + ', 2 component PCA, zbiór testowy')
    plt.legend(loc='lower left')
    plt.tight_layout()
    save_plot_as_png_file(plt)
    plt.show()

    count_print_confusion_matrix(X_train, X_test, y_train, y_test, lr)
    pass


url1 = "D:\\mgr\\heart-disease\\processed.switzerland.data"
url2 = "D:\\mgr\\heart-disease\\processed.cleveland.data"
url3 = "D:\\mgr\\heart-disease\\processed.hungarian.data"
url4 = "D:\\mgr\\heart-disease\\processed.va.data"

#process_pca(url1, 'Switzerland')
process_pca(url2, 'Cleveland')
#process_pca(url3, 'Hungarian')
#process_pca(url4, 'Long Beach, CA')
