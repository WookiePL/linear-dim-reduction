import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression

from reduction.results_metrics import count_print_confusion_matrix
from reduction.utils import plot_decision_regions, save_plot_as_png_file, standardise_classes


def process_lda(url, title):
    METHOD_NAME = 'LDA'
    # załadowanie zbioru danych do Pandas DataFrame
    df = pd.read_csv(url,
                     names=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
                            'slope', 'ca', 'thal', 'num, the predicted attribute'])

    df = df.replace('?', '0')
    # df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
    #               'ca', 'thal', 'num, the predicted attribute']
    df.convert_objects(convert_numeric=True)

    # podział na zbiór cech i klasy
    X, y = df.iloc[:, :13].values, df.iloc[:, 13].values

    y = standardise_classes(y)

    # podział danych na 70% zbiór treningowy, 30% testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # standaryzacja danych
    sc = StandardScaler()
    #sc = MinMaxScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    lda = LDA(n_components=2)
    X_train_lda = lda.fit_transform(X_train_std, y_train)
    X_test_lda = lda.transform(X_test_std)

    lr = LogisticRegression()
    lr = lr.fit(X_train_lda, y_train)

    count_print_confusion_matrix(X_train_lda, X_test_lda, y_train, y_test, lr)

    # plot_decision_regions(X_train_lda, y_train, classifier=lr, name="%s training" % title, method=METHOD_NAME)
    # plt.xlabel('LD 1')
    # plt.ylabel('LD 2')
    # plt.title(title + ', 2 component LDA, zbiór treningowy')
    # plt.legend(loc='lower left')
    # plt.tight_layout()
    # save_plot_as_png_file(plt)
    # plt.show()
    #
    #
    # plot_decision_regions(X_test_lda, y_test, classifier=lr, name="%s test" % title, method=METHOD_NAME)
    # plt.xlabel('LD 1')
    # plt.ylabel('LD 2')
    # plt.title(title + ', 2 component LDA, zbiór testowy')
    # plt.legend(loc='lower left')
    # plt.tight_layout()
    # save_plot_as_png_file(plt)
    # plt.show()


    pass


url1 = "D:\\mgr\\heart-disease\\processed.switzerland.data"
url2 = "D:\\mgr\\heart-disease\\processed.cleveland.data"
url3 = "D:\\mgr\\heart-disease\\processed.hungarian.data"
url4 = "D:\\mgr\\heart-disease\\processed.va.data"

#process_lda(url1, 'Switzerland')
process_lda(url2, 'Cleveland')
#process_lda(url3, 'Hungarian')
#process_lda(url4, 'Long Beach, CA')
