import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

def process_pca(url, title):
    # load dataset into Pandas DataFrame
    df = pd.read_csv(url, names=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num, the predicted attribute'])

    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num, the predicted attribute']
    df = df.replace('?', '0')
    df.convert_objects(convert_numeric=True)

    # Separating out the features
    x = df.loc[:, features].values

    # Separating out the target
    y = df.loc[:,['num, the predicted attribute']].values

    # Standardizing the features
    x = StandardScaler().fit_transform(x)


    pca = PCA(n_components=2)

    principalComponents = pca.fit_transform(x)

    principalDf = pd.DataFrame(data = principalComponents
                               , columns = ['principal component 1', 'principal component 2'])

    finalDf = pd.concat([principalDf, df[['num, the predicted attribute']]], axis = 1)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title(title + ', 2 component PCA', fontsize = 20)

    targets = [1, 2, 3]
    colors = ['r', 'g', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['num, the predicted attribute'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c = color
                   , s = 50)
    ax.legend(targets)
    ax.grid()
    plt.show()

url1 = "D:\\mgr\\heart-disease\\processed.switzerland.data"
url2 = "D:\\mgr\\heart-disease\\processed.cleveland.data"
url3 = "D:\\mgr\\heart-disease\\processed.hungarian.data"
url4 = "D:\\mgr\\heart-disease\\processed.va.data"


process_pca(url1, 'Switzerland')
process_pca(url2, 'Cleveland')
process_pca(url3, 'Hungarian')
process_pca(url4, 'Long Beach, CA')
