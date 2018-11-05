import os

from reduction.classifier_factory import ClassifierFactory
from reduction.utils import plot_decision_regions, save_plot_as_png_file
from reduction_dermatology.derm_utils import preprocess_dermatology_data
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from reduction_dermatology.results_metrics import count_print_confusion_matrix
from report_model.input_params import InputParams


def process_without_reduction(url, title, **kwargs):
    input_params = InputParams(os.path.basename(__file__), url, title, None, kwargs.get('classifier', 'lr'))
    method_name = 'Bez redukcji wymiarowości'
    X, y, X_train, X_test, y_train, y_test = preprocess_dermatology_data(url)


    # standaryzacja danych
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)



    # lr = LogisticRegression()
    # lr = lr.fit(X_train_std, y_train)

    _classifier = ClassifierFactory.get_classifier(kwargs)
    _classifier.fit(X_train_std, y_train)

    count_print_confusion_matrix(X_train_std, X_test_std, y_train, y_test, _classifier,
                                run_id=kwargs.get('run_id', '0'),
                                input_params=input_params)
    pass


url1 = "D:\\mgr\\dermatology\\dermatology.data"

# process_without_reduction(url1, 'Dermatology')
process_without_reduction(url1, 'Dermatology', classifier='tree')
