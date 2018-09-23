import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA


def show_intrinsic_dimensions(url, title):
    # load dataset into Pandas DataFrame
    df = pd.read_csv(url,
                     names=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
                            'slope', 'ca', 'thal'])

    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
                'thal']
    df = df.replace('?', '0')
    df.convert_objects(convert_numeric=True)

    # normalize data
    data_scaled = pd.DataFrame(preprocessing.scale(df), columns=df.columns)

    pca = PCA()
    pca.fit(df)

    features2 = range(pca.n_components_)

    plt.bar(features2, pca.explained_variance_)
    plt.xticks(features2)
    plt.ylabel('variance')
    plt.xlabel('PCA feature')
    plt.show()

    indexes = []
    for i in range(13):
        indexes.append('PC-' + (1 + i).__str__())

    pd.set_option('display.max_columns', None)
    print(pd.DataFrame(pca.components_, columns=data_scaled.columns, index=indexes))


url1 = "D:\\mgr\\heart-disease\\processed.switzerland.data"
url2 = "D:\\mgr\\heart-disease\\processed.cleveland.data"
url3 = "D:\\mgr\\heart-disease\\processed.hungarian.data"
url4 = "D:\\mgr\\heart-disease\\processed.va.data"

show_intrinsic_dimensions(url1, 'Switzerland')
show_intrinsic_dimensions(url2, 'Cleveland')
show_intrinsic_dimensions(url3, 'Hungarian')
show_intrinsic_dimensions(url4, 'Long Beach, CA')
