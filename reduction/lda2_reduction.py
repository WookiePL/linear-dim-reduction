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
from report_model.input_params import InputParams
from reduction.classifier_factory import ClassifierFactory
import os


def process_lda(url, title, n_components, **kwargs):
    input_params = InputParams(os.path.basename(__file__), url, title, n_components, kwargs.get('classifier', 'lr'))
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

    lda = LDA(n_components=n_components)
    X_train_lda = lda.fit_transform(X_train_std, y_train)
    X_test_lda = lda.transform(X_test_std)

    # lr = LogisticRegression()
    # lr = lr.fit(X_train_lda, y_train)
    _classifier = ClassifierFactory.get_classifier(kwargs)
    _classifier.fit(X_train_lda, y_train)

    count_print_confusion_matrix(X_train_lda, X_test_lda, y_train, y_test, _classifier,
                                 run_id=kwargs.get('run_id', '0'),
                                 input_params=input_params)

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


url1 = "F:\\mgr\\heart-disease\\processed.switzerland.data"
url2 = "F:\\mgr\\heart-disease\\processed.cleveland.data"
url3 = "F:\\mgr\\heart-disease\\processed.hungarian.data"
url4 = "F:\\mgr\\heart-disease\\processed.va.data"

#process_lda(url1, 'Switzerland', 2, classifier='lr')
# process_lda(url2, 'Cleveland', 1, classifier='lr')
# process_lda(url2, 'Cleveland', 8, classifier='lr')
#process_lda(url3, 'Hungarian', 2)
#process_lda(url4, 'Long Beach, CA', 2)
