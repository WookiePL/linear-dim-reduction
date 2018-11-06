import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, normalize
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression

from reduction.utils import save_plot_as_png_file, standardise_classes, plot_decision_regions


def process_nmf(url, title):
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
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)


    cca = CCA(n_components=2)
    X_train_pca = cca.fit_transform(X_train_std)
    X_test_pca = cca.transform(X_test_std)

    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1])
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.show()


    lr = LogisticRegression()
    lr = lr.fit(X_train_pca, y_train)


    plot_decision_regions(X_train_pca, y_train, classifier=lr, name="%s training" % title)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title(title + ', 2 component NMF, zbiór treningowy')
    plt.legend(loc='lower left')
    plt.tight_layout()
    save_plot_as_png_file(plt)
    plt.show()

    plot_decision_regions(X_test_pca, y_test, classifier=lr, name="%s test" % title)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title(title + ', 2 component NMF, zbiór testowy')
    plt.legend(loc='lower left')
    plt.tight_layout()
    save_plot_as_png_file(plt)
    plt.show()
    pass



url1 = "F:\\mgr\\dermatology\\dermatology.data"

process_nmf(url1, 'Dermatology')
