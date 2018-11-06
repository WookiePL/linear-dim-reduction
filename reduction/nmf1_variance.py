import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from reduction.results_metrics import count_print_confusion_matrix
from reduction.utils import save_plot_as_png_file, standardise_classes, plot_decision_regions
from report_model.input_params import InputParams
from reduction.classifier_factory import ClassifierFactory
import os
import numpy as np


def process_nmf(url, title, n_components, **kwargs):
    input_params = InputParams(os.path.basename(__file__), url, title, n_components, kwargs.get('classifier', 'lr'))

    METHOD_NAME = 'NMF'

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
    # TODO zrobic dobra standatyzacje danych dla NMF
    # sc = MaxAbsScaler()
    if title in ('Hungarian', 'Cleveland'):
        sc = StandardScaler(with_mean=False)
    else:
        sc = MinMaxScaler(feature_range=(1, 2))
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)


    # otrzymywanie wartości własnych (eigenvalues)
    cov_mat = np.cov(X_train_std.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    print('Eigenvalues \n%s' % eigen_vals)

    # suma wariancji
    total = sum(eigen_vals)
    # wariancje wyjaśnioen
    var_exp = [(i / total) for i in sorted(eigen_vals, reverse=True)]
    # łączna suma wariancji wyjaśnionej
    cum_var_exp = np.cumsum(var_exp)

    plt.bar(range(1, 14), var_exp, alpha=0.5, align='center',
            label='pojedyncza wariancja wyjaśniona')
    plt.step(range(1, 14), cum_var_exp, where='mid',
             label='łączna wariancja wyjaśniona')
    plt.ylabel('współczynnik wariancji wyjaśnionej')
    plt.xlabel('główne składowe')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


    pass


# url1 = "F:\\mgr\\heart-disease\\processed.switzerland.data"
url2 = "F:\\mgr\\heart-disease\\processed.cleveland.data"
# url3 = "F:\\mgr\\heart-disease\\processed.hungarian.data"
# url4 = "F:\\mgr\\heart-disease\\processed.va.data"
#
# process_nmf(url1, 'Switzerland', 2)
process_nmf(url2, 'Cleveland', 2)
# process_nmf(url3, 'Hungarian', 2)
# process_nmf(url4, 'Long Beach, CA', 2)
