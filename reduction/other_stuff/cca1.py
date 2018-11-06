import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from reduction.utils import save_plot_as_png_file, standardise_classes, plot_decision_regions


def process_pca(url, title):
    # załadowanie zbioru danych do Pandas DataFrame
    df = pd.read_csv(url,
                     names=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
                            'slope', 'ca', 'thal', 'num, the predicted attribute'])

    df = df.replace('?', '0')

    df.convert_objects(convert_numeric=True)
    print(df.dtypes)

    # podział na zbiór cech i klasy
    X, y = df.iloc[:, :13].values, df.iloc[:, 13].values

    y = standardise_classes(y)

    # podział danych na 70% zbiór treningowy, 30% testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # standaryzacja danych
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    cca = CCA(n_components=2)
    X_train_pca = cca.fit_transform(X_train_std)
    X_test_pca = cca.transform(X_test_std)

    pass

url1 = "F:\\mgr\\heart-disease\\processed.switzerland.data"
url2 = "F:\\mgr\\heart-disease\\processed.cleveland.data"
url3 = "F:\\mgr\\heart-disease\\processed.hungarian.data"
url4 = "F:\\mgr\\heart-disease\\processed.va.data"

#process_pca(url1, 'Switzerland')
process_pca(url2, 'Cleveland')
# process_pca(url3, 'Hungarian')
# process_pca(url4, 'Long Beach, CA')
