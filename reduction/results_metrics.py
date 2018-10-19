import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, cohen_kappa_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def count_print_confusion_matrix(X_train, X_test, y_train, y_test, classifier):

    # TO JEST DLA NIEZREDUKOWANEGO:
    # pipe_svc = Pipeline([('scl', StandardScaler()), ('clf', LogisticRegression())])
    # pipe_svc.fit(X_train, y_train)

    #TO JEST DLA ZREDUKOWANEGO
    y_pred = classifier.predict(X_test)
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print(conf_matrix)

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,
                    s=conf_matrix[i, j],
                    va='center', ha='center')
    plt.xlabel('przewidywana klasa')
    plt.ylabel('rzeczywista klasa')
    plt.show()

    print('Wynik F1: %.4f' % f1_score(y_true=y_test, y_pred=y_pred))
    print('Dokładność: %.4f' % accuracy_score(y_true=y_test, y_pred=y_pred))
    print('Cohen’s kappa: %.4f' % cohen_kappa_score(y_test, y_pred))

# def count_f1(X_train, X_test, y_train, y_test):
#     print('wynik F1: %s' % f1_score(y_true=y_test, y_pred=y_pred))
