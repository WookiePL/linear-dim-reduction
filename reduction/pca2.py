import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

def show_intrinsic_dimensions(url, title):
    # load dataset into Pandas DataFrame
    df = pd.read_csv(url, names=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    df = df.replace('?', '0')
    df.convert_objects(convert_numeric=True)


    pca = PCA()

    pca.fit(df)

    features2 = range(pca.n_components_)

    plt.bar(features2, pca.explained_variance_)
    plt.xticks(features2)
    plt.ylabel('variance')
    plt.xlabel('PCA feature')
    plt.show()


url1 = "D:\\mgr\\heart-disease\\processed.switzerland.data"
url2 = "D:\\mgr\\heart-disease\\processed.cleveland.data"
url3 = "D:\\mgr\\heart-disease\\processed.hungarian.data"
url4 = "D:\\mgr\\heart-disease\\processed.va.data"


show_intrinsic_dimensions(url1, 'Switzerland')
show_intrinsic_dimensions(url2, 'Cleveland')
show_intrinsic_dimensions(url3, 'Hungarian')
show_intrinsic_dimensions(url4, 'Long Beach, CA')
