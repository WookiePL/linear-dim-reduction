import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, cohen_kappa_score, classification_report, \
    r2_score, mean_absolute_error, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

from reduction_dermatology.results_metrics import generate_output_report


def count_print_confusion_matrix(X_train, X_test, y_train, y_test, classifier, **kwargs):

    # TO JEST DLA NIEZREDUKOWANEGO:
    # pipe_svc = Pipeline([('scl', StandardScaler()), ('clf', LogisticRegression())])
    # pipe_svc.fit(X_train, y_train)

    #TO JEST DLA ZREDUKOWANEGO
    y_pred = classifier.predict(X_test)
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print(conf_matrix)

    plot_confusion_matrix(conf_matrix)

    print('Wynik F1: %.4f' % f1_score(y_true=y_test, y_pred=y_pred))
    print('Accuracy: %.4f' % accuracy_score(y_true=y_test, y_pred=y_pred))
    print('Cohen’s kappa: %.4f' % cohen_kappa_score(y_test, y_pred))
    print('R2 Score: %.4f' % r2_score(y_true=y_test, y_pred=y_pred))

    #https://stackoverflow.com/questions/33275461/specificity-in-scikit-learn
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp)
    print('Specificity: %.4f' % specificity)

    target_names = ['class 0', 'class 1']
    print(classification_report(y_true=y_test, y_pred=y_pred,  target_names=target_names))

    _f1_score = f1_score(y_true=y_test, y_pred=y_pred, average='binary')
    _error = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    _accuracy_score = accuracy_score(y_true=y_test, y_pred=y_pred)
    _cohen_kappa_score = cohen_kappa_score(y_test, y_pred)
    #r2_score(y_true=y_test, y_pred=y_pred)
    _precision_score = precision_score(y_true=y_test, y_pred=y_pred, average='binary')
    _recall_score = recall_score(y_true=y_test, y_pred=y_pred, average='binary')
    generate_output_report(_accuracy_score, _error, _f1_score, _precision_score, _recall_score, conf_matrix, kwargs)
    pass


def plot_confusion_matrix(conf_matrix):
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Greys, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,
                    s=conf_matrix[i, j],
                    va='center', ha='center')
    plt.xlabel('przewidywana klasa')
    plt.ylabel('rzeczywista klasa')
    plt.show()


def desc_confusion_matrix():
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    conf_matrix = np.array([['PP', 'FN'], ['FP', 'PN']])
    #ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    ax.matshow(conf_matrix, cmap=plt.cm.Greys, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,
                    s=conf_matrix[i],
                    va='center', ha='center')
    # ax.text(x=0, y=0, s=conf_matrix[0], va='center', ha='center')
    # ax.text(x=0, y=1, s=conf_matrix[1], va='center', ha='center')
    # ax.text(x=1, y=0, s=conf_matrix[2], va='center', ha='center')
    # ax.text(x=1, y=1, s=conf_matrix[3], va='center', ha='center')
    plt.xlabel('przewidywana klasa')
    plt.ylabel('rzeczywista klasa')
    plt.show()


#desc_confusion_matrix()
# def count_f1(X_train, X_test, y_train, y_test):
#     print('wynik F1: %s' % f1_score(y_true=y_test, y_pred=y_pred))
