import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, cohen_kappa_score, classification_report, \
    r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

def count_print_confusion_matrix(X_train, X_test, y_train, y_test, classifier):

    # TO JEST DLA NIEZREDUKOWANEGO:
    # pipe_svc = Pipeline([('scl', StandardScaler()), ('clf', LogisticRegression())])
    # pipe_svc.fit(X_train, y_train)

    #TO JEST DLA ZREDUKOWANEGO
    y_pred = classifier.predict(X_test)
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print(conf_matrix)

    # fig, ax = plt.subplots(figsize=(2.5, 2.5))
    # ax.matshow(conf_matrix, cmap=plt.cm.Greys, alpha=0.3)
    # for i in range(conf_matrix.shape[0]):
    #     for j in range(conf_matrix.shape[1]):
    #         ax.text(x=j, y=i,
    #                 s=conf_matrix[i, j],
    #                 va='center', ha='center')
    # plt.xlabel('przewidywana klasa')
    # plt.ylabel('rzeczywista klasa')
    # plt.show()

    # print('Wynik F1: %.4f' % f1_score(y_true=y_test, y_pred=y_pred))
    print('Accuracy: %.4f' % accuracy_score(y_true=y_test, y_pred=y_pred))
    print('Cohen’s kappa: %.4f' % cohen_kappa_score(y_test, y_pred))
    print('R2 Score: %.4f' % r2_score(y_true=y_test, y_pred=y_pred))

    #https://stackoverflow.com/questions/33275461/specificity-in-scikit-learn
    # tn, fp, fn, tp = conf_matrix.ravel()
    # specificity = tn / (tn + fp)
    # print('Specificity: %.4f' % specificity)
    #
    target_names = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6']
    print(classification_report(y_true=y_test, y_pred=y_pred,  target_names=target_names))
    pass