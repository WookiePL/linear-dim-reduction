import matplotlib.pyplot as plt
import pandas as pd
from sklearn import cross_validation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
import numpy as np

from reduction.utils import save_plot_as_png_file


def process_pca(url, title):
    # load dataset into Pandas DataFrame
    df = pd.read_csv(url,
                     names=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
                            'slope', 'ca', 'thal', 'num, the predicted attribute'])

    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
                'thal']
    df = df.replace('?', '0')
    df.convert_objects(convert_numeric=True)

    # Separating out the features, we will be calling them x
    x = df.loc[:, features].values

    # Separating out the target, we will be calling them y
    y = df.loc[:, ['num, the predicted attribute']].values

    for index, item in enumerate(y):
        if not (item == 0):
            y[index] = 1

    # Standardizing the features
    x = StandardScaler().fit_transform(x)

    # applying PCA
    pca = PCA(n_components=2)
    x_principalComponents = pca.fit_transform(x)

    principalDf = pd.DataFrame(data=x_principalComponents
                               , columns=['principal component 1', 'principal component 2'])

    # normalize data, replace >1 values with 1 TODO: why this is done 2 times?
    classes = df['num, the predicted attribute']
    for index, item in enumerate(classes):
        if not (item == 0):
            classes[index] = 1
    df[['num, the predicted attribute']] = classes

    final_df = pd.concat([principalDf, classes], axis=1)

    # plot the graph
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title(title + ', 2 component PCA', fontsize=20)

    targets = [0, 1, 2, 3, 4]
    # colors = ['r', 'y', 'g', 'b', 'm']
    colors = ['r', 'g', 'b', 'm', 'y']
    for target, color in zip(targets, colors):
        indicesToKeep = final_df['num, the predicted attribute'] == target
        ax.scatter(final_df.loc[indicesToKeep, 'principal component 1']
                   , final_df.loc[indicesToKeep, 'principal component 2']
                   , c=color
                   , s=50)
    ax.legend(targets)
    ax.grid()
    plt.tight_layout()
    save_plot_as_png_file(plt)
    plt.show()

    # SVM Support Vector Machine
    modelSVM = LinearSVC(C=0.001)
    # cross validation on the training and test sets
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_principalComponents, y, test_size=0.2,
                                                                         train_size=0.8, random_state=0)
    modelSVM = modelSVM.fit(x_train, y_train)
    print("Linear Support Vector Machine (SVM) values with split:")
    print(modelSVM.score(x_test, y_test))

    modelSVMRaw = LinearSVC(C=0.001)
    modelSVMRaw = modelSVMRaw.fit(x_principalComponents, y)
    counter = 0
    for index in modelSVMRaw.predict(x_principalComponents):
        if index == y[index]:
            counter = counter + 1
    print("Score without any split:")
    print(float(counter) / 303)

    modelSVM2 = SVC(C=0.001, kernel='rbf')
    x_train2, x_test2, y_train2, y_test2 = cross_validation.train_test_split(x_principalComponents, y, test_size=0.2,
                                                                             train_size=0.8, random_state=0)

    modelSVM2 = modelSVM2.fit(x_train2, y_train2)
    print("RBF score with split:")
    print(modelSVM2.score(x_test2, y_test2))
    counter2 = 0
    for index in modelSVMRaw.predict(x_principalComponents):
        if index == y[index]:
            counter2 = counter2 + 1
    print("RBF score without any split:")
    print(float(counter2) / 303)



    # x_min, x_max = x_principalComponents[:, 0].min() - 1, x_principalComponents[:, 0].max() + 1
    # y_min, y_max = x_principalComponents[:, 1].min() - 1, x_principalComponents[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
    #                      np.arange(y_min, y_max, 0.02))
    #
    # # title for the plots
    # titles = 'SVC (RBF kernel)- Plotting highest varied 2 PCA values'
    #
    #
    # # Plot the decision boundary. For that, we will assign a color to each
    # # point in the mesh [x_min, m_max]x[y_min, y_max].
    # plt.subplot(2, 2, 2 + 1)
    # plt.subplots_adjust(wspace=0.4, hspace=0.4)
    # Z = modelSVM2.predict(np.c_[xx.ravel(), yy.ravel()])
    #
    # # Put the result into a color plot
    # Z = Z.reshape(xx.shape)
    # plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    # # Plot also the training points
    # plt.scatter(x_principalComponents[:, 0], x_principalComponents[:, 1], c=y, cmap=plt.cm.Paired)
    # plt.xlabel('PCA 1')
    # plt.ylabel('PCA 2')
    # plt.xlim(xx.min(), xx.max())
    # plt.ylim(yy.min(), yy.max())
    # plt.xticks(())
    # plt.yticks(())
    # plt.title(titles)
    # plt.show()


url1 = "D:\\mgr\\heart-disease\\processed.switzerland.data"
url2 = "D:\\mgr\\heart-disease\\processed.cleveland.data"
url3 = "D:\\mgr\\heart-disease\\processed.hungarian.data"
url4 = "D:\\mgr\\heart-disease\\processed.va.data"

process_pca(url1, 'Switzerland')
process_pca(url2, 'Cleveland')
process_pca(url3, 'Hungarian')
process_pca(url4, 'Long Beach, CA')
