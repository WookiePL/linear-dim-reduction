import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from reduction.utils import plot_decision_regions, save_plot_as_png_file, standardise_classes


def process_lda(url, title):
    method_name = 'LDA'
    # załadowanie zbioru danych do Pandas DataFrame
    df = pd.read_csv(url,
                     names=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
                            'slope', 'ca', 'thal', 'num, the predicted attribute'])

    df = df.replace('?', '0')
    df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
                  'ca', 'thal', 'num, the predicted attribute']
    df.convert_objects(convert_numeric=True)

    # podział na zbiór cech i klasy
    X, y = df.iloc[:, :13].values, df.iloc[:, 13].values

    y = standardise_classes(y)

    # podział danych na 70% zbiór treningowy, 30% testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # standaryzacja danych
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    np.set_printoptions(precision=4) #ilosc miejsc po przecinku dla liczb w macierzach
    mean_vectors = []
    for label in range(1, 2):
        mean_vectors.append(np.mean(X_train_std[y_train == label], axis=0))
        print('MeanVector %s: %s\n' % (label, mean_vectors[label - 1]))

    # liczba cech
    d = 13
    S_W = np.zeros((d, d))
    for label, mv in zip(range(1, 3), mean_vectors):
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



    d = 13  # number of features
    S_W = np.zeros((d, d))
    for label, mv in zip(range(1, 3), mean_vectors):
        class_scatter = np.cov(X_train_std[y_train == label].T)
        S_W += class_scatter
    print('Scaled within-class scatter matrix: %sx%s' % (S_W.shape[0],
                                                         S_W.shape[1]))


    mean_overall = np.mean(X_train_std, axis=0)
    d = 13  # number of features
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

    plt.bar(range(1, 14), discr, alpha=0.5, align='center',
            label='individual "discriminability"')
    plt.step(range(1, 14), cum_discr, where='mid',
             label='cumulative "discriminability"')
    plt.ylabel('"discriminability" ratio')
    plt.xlabel('Linear Discriminants')
    plt.ylim([-0.1, 1.1])
    plt.legend(loc='best')
    plt.tight_layout()
    # plt.savefig('./figures/lda1.png', dpi=300)
    plt.show()

   #  w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
   #                 eigen_pairs[1][1][:, np.newaxis].real))
   #  print('Matrix W:\n', w)
   #
   #
   #
   #
   #  X_train_lda = X_train_std.dot(w)
   #  colors = ['r', 'b', 'g']
   #  markers = ['s', 'x', 'o']
   # # markers = ('s', 'x', 'o', '^', 'v')
   # # colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
   #
   #  for l, c, m in zip(np.unique(y_train), colors, markers):
   #      plt.scatter(X_train_lda[y_train == l, 0] * (-1),
   #                  X_train_lda[y_train == l, 1] * (-1),
   #                  c=c, label=l, marker=m)
   #
   #  plt.xlabel('LD 1')
   #  plt.ylabel('LD 2')
   #  plt.legend(loc='lower right')
   #  plt.tight_layout()
   #  # plt.savefig('./figures/lda2.png', dpi=300)
   #  plt.show()
   #
   #
   #  from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
   #
   #  lda = LDA(n_components=2)
   #  X_train_lda = lda.fit_transform(X_train_std, y_train)
   #
   #  from sklearn.linear_model import LogisticRegression
   #  lr = LogisticRegression()
   #  lr = lr.fit(X_train_lda, y_train)
   #
   #  # REMOVE THIS STUFF:
   #  # pred_train = lr.predict(X_train_lda)
   #  # print('\nPrediction accuracy for the normal test dataset with PCA')
   #  # print('{:.2%}\n'.format(metrics.accuracy_score(y_train, pred_train)))
   #
   #  plot_decision_regions(X_train_lda, y_train, classifier=lr, name="%s training" % title, method=method_name)
   #  plt.xlabel('LD 1')
   #  plt.ylabel('LD 2')
   #  plt.title(title + ', 2 component LDA, zbiór treningowy')
   #  plt.legend(loc='lower left')
   #  plt.tight_layout()
   #  save_plot_as_png_file(plt)
   #  plt.show()
   #
   #  X_test_lda = lda.transform(X_test_std)
   #
   #  plot_decision_regions(X_test_lda, y_test, classifier=lr, name="%s test" % title, method=method_name)
   #  plt.xlabel('LD 1')
   #  plt.ylabel('LD 2')
   #  plt.title(title + ', 2 component LDA, zbiór testowy')
   #  plt.legend(loc='lower left')
   #  plt.tight_layout()
   #  save_plot_as_png_file(plt)
   #  plt.show()


    pass


url1 = "F:\\mgr\\heart-disease\\processed.switzerland.data"
url2 = "F:\\mgr\\heart-disease\\processed.cleveland.data"
url3 = "F:\\mgr\\heart-disease\\processed.hungarian.data"
url4 = "F:\\mgr\\heart-disease\\processed.va.data"

# process_lda(url1, 'Switzerland')
process_lda(url2, 'Cleveland')
# process_lda(url3, 'Hungarian')
# process_lda(url4, 'Long Beach, CA')
