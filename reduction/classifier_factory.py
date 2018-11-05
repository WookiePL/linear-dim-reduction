from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


class ClassifierFactory:
    def __init__(self):
        pass

    @classmethod
    def get_classifier(cls, kwargs):
        classifier = kwargs.get('classifier', 'lr')

        if classifier == 'lr':
            return cls.__get_logistic_regression()
        elif classifier == 'svm':
            return cls.__get_support_vector_machine()
        elif classifier == 'tree':
            return cls.__get_decision_tree()
        elif classifier == 'forest':
            return cls.__get_random_forest()

    @staticmethod
    def __get_logistic_regression():
        return LogisticRegression()

    @staticmethod
    def __get_support_vector_machine():
        return SVC(kernel='rbf', random_state=0, gamma=0.7, C=1.0)

    @staticmethod
    def __get_decision_tree():
        return DecisionTreeClassifier(criterion='entropy',
                                      max_depth=3,
                                      random_state=0)

    @staticmethod
    def __get_random_forest():
        return RandomForestClassifier(criterion='entropy',
                                      n_estimators=10,
                                      random_state=1,
                                      n_jobs=2)
