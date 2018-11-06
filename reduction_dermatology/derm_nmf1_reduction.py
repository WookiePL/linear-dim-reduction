import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, normalize, scale
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression

from reduction.classifier_factory import ClassifierFactory
from reduction.utils import save_plot_as_png_file, standardise_classes, plot_decision_regions, \
    plot_decision_regions_for_nmf
from reduction_dermatology.results_metrics import count_print_confusion_matrix
from report_model.input_params import InputParams
import os


def process_nmf(url, title, n_components, **kwargs):
    input_params = InputParams(os.path.basename(__file__), url, title, n_components, kwargs.get('classifier', 'lr'))

    METHOD_NAME='NMF'
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

   # y = standardise_classes(y)

    #podział danych na 70% zbiór treningowy, 30% testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    #standaryzacja danych
    #sc = MaxAbsScaler()
    # sc = StandardScaler(with_mean = False)
    sc = MinMaxScaler(feature_range=(0, 3))
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


    nmf = NMF(n_components=n_components)
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

    if n_components == 2:
        plot_decision_regions_for_nmf(X_train_nmf, y_train, classifier=_classifier, name="%s training" % title, method=METHOD_NAME)
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.title(title + ', 2 component NMF, zbiór treningowy')
        plt.legend(loc='lower left')
        plt.tight_layout()
        save_plot_as_png_file(plt)
        plt.show()

        plot_decision_regions_for_nmf(X_test_nmf, y_test, classifier=_classifier, name="%s test" % title, method=METHOD_NAME)
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.title(title + ', 2 component NMF, zbiór testowy')
        plt.legend(loc='lower left')
        plt.tight_layout()
        save_plot_as_png_file(plt)
        plt.show()

    count_print_confusion_matrix(X_train_nmf, X_test_nmf, y_train, y_test, _classifier,
                                 run_id=kwargs.get('run_id', '0'),
                                 input_params=input_params,
                                 training_png_url=training_png_url,
                                 test_png_url=test_png_url)
    pass



url1 = "F:\\mgr\\dermatology\\dermatology.data"

process_nmf(url1, 'Dermatology', n_components=2)
