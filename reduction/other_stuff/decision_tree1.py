import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from reduction.utils import save_plot_as_png_file, standardise_classes, plot_decision_regions, \
    preprocess_heart_disease_data


def process_decision_tree(url, title):

    X, y, X_train, X_test, y_train, y_test = preprocess_heart_disease_data(url)

    tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
    tree.fit(X_train, y_train)

    plot_decision_regions(X_train, y_train, classifier=tree, name="%s training" % title)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title(title + ', 2 component Kernel PCA, zbiór treningowy')
    plt.legend(loc='lower left')
    plt.tight_layout()
    save_plot_as_png_file(plt)
    plt.show()

url1 = "F:\\mgr\\heart-disease\\processed.switzerland.data"
url2 = "F:\\mgr\\heart-disease\\processed.cleveland.data"
url3 = "F:\\mgr\\heart-disease\\processed.hungarian.data"
url4 = "F:\\mgr\\heart-disease\\processed.va.data"

#process_decision_tree(url1, 'Switzerland')
process_decision_tree(url2, 'Cleveland')
#process_decision_tree(url3, 'Hungarian')
# process_decision_tree(url4, 'Long Beach, CA')