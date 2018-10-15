import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, normalize
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression

from reduction.utils import save_plot_as_png_file, standardise_classes, plot_decision_regions


def process_nmf(url, title):
    METHOD_NAME='NMF'

    # załadowanie zbioru danych do Pandas DataFrame
    df = pd.read_csv(url,
                     names=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
                            'slope', 'ca', 'thal', 'num, the predicted attribute'])

    df = df.replace('?', '0')
    df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num, the predicted attribute']
    df.convert_objects(convert_numeric=True)

    #podział na zbiór cech i klasy
    X, y = df.iloc[:, :13].values, df.iloc[:, 13].values

    y = standardise_classes(y)

    #podział danych na 70% zbiór treningowy, 30% testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    #standaryzacja danych
    #TODO zrobic dobra standatyzacje danych dla NMF
    #sc = MaxAbsScaler()
    sc = StandardScaler(with_mean = False)
    #sc = MinMaxScaler(feature_range=(0, 1))
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)


    nmf = NMF(n_components=2)
    X_train_pca = nmf.fit_transform(X_train_std)
    X_test_pca = nmf.transform(X_test_std)

    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1])
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.show()

    # def plot_decision_regions(X, y, classifier, resolution=0.02):
    # # setup marker generator and color map
    #     markers = ('s', 'x', 'o', '^', 'v')
    #     colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    #     cmap = ListedColormap(colors[:len(np.unique(y))])
    #
    #     # plot the decision surface
    #     x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    #     x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    #     xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    #                            np.arange(x2_min, x2_max, resolution))
    #     Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    #     Z = Z.reshape(xx1.shape)
    #     plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    #     plt.xlim(xx1.min(), xx1.max())
    #     plt.ylim(xx2.min(), xx2.max())
    #
    #     # plot class samples
    #     for idx, cl in enumerate(np.unique(y)):
    #         plt.scatter(x=X[y == cl, 0],
    #                     y=X[y == cl, 1],
    #                     alpha=0.6,
    #                     c=cmap(idx),
    #                     edgecolor='black',
    #                     marker=markers[idx],
    #                     label=cl)

    lr = LogisticRegression()
    lr = lr.fit(X_train_pca, y_train)


    plot_decision_regions(X_train_pca, y_train, classifier=lr, name="%s test" % title, method=METHOD_NAME)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title(title + ', 2 component NMF, zbiór treningowy')
    plt.legend(loc='lower left')
    plt.tight_layout()
    save_plot_as_png_file(plt)
    plt.show()

    plot_decision_regions(X_test_pca, y_test, classifier=lr, name="%s trening" % title, method=METHOD_NAME)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title(title + ', 2 component NMF, zbiór testowy')
    plt.legend(loc='lower left')
    plt.tight_layout()
    save_plot_as_png_file(plt)
    plt.show()
    pass



url1 = "D:\\mgr\\heart-disease\\processed.switzerland.data"
url2 = "D:\\mgr\\heart-disease\\processed.cleveland.data"
url3 = "D:\\mgr\\heart-disease\\processed.hungarian.data"
url4 = "D:\\mgr\\heart-disease\\processed.va.data"

#process_nmf(url1, 'Switzerland')
process_nmf(url2, 'Cleveland')
#process_pca(url3, 'Hungarian')
#process_pca(url4, 'Long Beach, CA')
