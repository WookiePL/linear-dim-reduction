import itertools
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, cohen_kappa_score, classification_report, \
    r2_score, recall_score, precision_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

from reduction.utils import get_name_for_json_file, get_run_id
from report_model.output_report import OutputReport


def count_print_confusion_matrix(X_train, X_test, y_train, y_test, classifier, **kwargs):
    target_names = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6']
    # TO JEST DLA NIEZREDUKOWANEGO:
    # pipe_svc = Pipeline([('scl', StandardScaler()), ('clf', LogisticRegression())])
    # pipe_svc.fit(X_train, y_train)

    #TO JEST DLA ZREDUKOWANEGO
    y_pred = classifier.predict(X_test)
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print(conf_matrix)
    plot_confusion_matrix(conf_matrix, target_names, title='Macierz pomyłek')

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
    _f1_score = f1_score(y_true=y_test, y_pred=y_pred, average='micro')
    _error = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    _accuracy_score = accuracy_score(y_true=y_test, y_pred=y_pred)
    _cohen_kappa_score = cohen_kappa_score(y_test, y_pred)
    #r2_score(y_true=y_test, y_pred=y_pred)
    _precision_score = precision_score(y_true=y_test, y_pred=y_pred, average='micro')
    _recall_score = recall_score(y_true=y_test, y_pred=y_pred, average='micro')

    #print('Wynik F1: %.4f' % f1_score(y_true=y_test, y_pred=y_pred))
    print('Accuracy: %.4f' % accuracy_score(y_true=y_test, y_pred=y_pred))
    print('Cohen’s kappa: %.4f' % cohen_kappa_score(y_test, y_pred))
    #print('R2 Score: %.4f' % r2_score(y_true=y_test, y_pred=y_pred))

    #https://stackoverflow.com/questions/33275461/specificity-in-scikit-learn
    # tn, fp, fn, tp = conf_matrix.ravel()
    # specificity = tn / (tn + fp)
    # print('Specificity: %.4f' % specificity)
    #

    report = classification_report(y_true=y_test, y_pred=y_pred, target_names=target_names)
    print(report)

    generate_output_report(_accuracy_score, _error, _f1_score, _precision_score, _recall_score, conf_matrix, kwargs)

    pass


def generate_output_report(_accuracy_score, _error, _f1_score, _precision_score, _recall_score, conf_matrix, kwargs):
    run_id = kwargs.get('run_id')
    input_params = kwargs.get('input_params')
    training_png_url = kwargs.get('training_png_url')
    test_png_url = kwargs.get('test_png_url')
    output_report = OutputReport(run_id=run_id,
                                 input_params=input_params,
                                 plot_training_png_url=training_png_url,
                                 plot_test_png_url=test_png_url,
                                 f1_score=_f1_score,
                                 error=_error,
                                 accuracy=_accuracy_score,
                                 precision=_precision_score,
                                 recall=_recall_score,
                                 conf_matrix=conf_matrix.tolist(),
                                 conf_matrix_png_url='')
    # with open('output_report.json', 'w') as outfile:
    #     json.dump(output_report.__dict__, outfile)
    # https://stackoverflow.com/questions/3768895/how-to-make-a-class-json-serializable
    output_json = output_report.toJSON()
    print(output_json)
    output_json_file = open(get_name_for_json_file(), mode='w')
    output_json_file.write(output_json)
    output_json_file.close()


def plot_confusion_matrix(conf_matrix, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greys):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(conf_matrix)

    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

    plt.xlabel('przewidywana klasa')
    plt.ylabel('rzeczywista klasa')
    plt.tight_layout()
    plt.show()
