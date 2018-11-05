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

    nmf = NMF(n_components=2)
    X_train_nmf = nmf.fit_transform(X_train_std)
    X_test_nmf = nmf.transform(X_test_std)

    plt.scatter(X_train_nmf[:, 0], X_train_nmf[:, 1])
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.show()

    # lr = LogisticRegression()
    # lr = lr.fit(X_train_nmf, y_train)
    _classifier = ClassifierFactory.get_classifier(kwargs)
    _classifier.fit(X_train_nmf, y_train)

    training_png_url = ''
    test_png_url = ''

    if n_components == 2: # jesli 2 wymiary to mozna narysowac wykres liniowy
        plot_decision_regions(X_train_nmf, y_train, classifier=_classifier, name="%s test" % title, method=METHOD_NAME)
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.title(title + ', 2 component NMF, zbiór treningowy')
        plt.legend(loc='lower left')
        plt.tight_layout()
        training_png_url = save_plot_as_png_file(plt)
        plt.show()

        plot_decision_regions(X_test_nmf, y_test, classifier=_classifier, name="%s trening" % title, method=METHOD_NAME)
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.title(title + ', 2 component NMF, zbiór testowy')
        plt.legend(loc='lower left')
        plt.tight_layout()
        test_png_url = save_plot_as_png_file(plt)
        plt.show()

    count_print_confusion_matrix(X_train_nmf, X_test_nmf, y_train, y_test, _classifier,
                                 run_id=kwargs.get('run_id', '0'),
                                 input_params=input_params,
                                 training_png_url=training_png_url,
                                 test_png_url=test_png_url)
    pass


url1 = "D:\\mgr\\heart-disease\\processed.switzerland.data"
url2 = "D:\\mgr\\heart-disease\\processed.cleveland.data"
url3 = "D:\\mgr\\heart-disease\\processed.hungarian.data"
url4 = "D:\\mgr\\heart-disease\\processed.va.data"

process_nmf(url1, 'Switzerland', 2)
process_nmf(url2, 'Cleveland', 2)
process_nmf(url3, 'Hungarian', 2)
process_nmf(url4, 'Long Beach, CA', 2)
