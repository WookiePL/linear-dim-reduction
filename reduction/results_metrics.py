import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def count_print_confusion_matrix(X_train, X_test, y_train, y_test, classifier):
    pipe_svc = Pipeline([('scl', StandardScaler()), ('clf', SVC(random_state=1))])
    pipe_svc.fit(X_train, y_train)
    y_pred = pipe_svc.predict(X_test)
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
