import multiprocessing

from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from reduction_dermatology.derm_utils import preprocess_dermatology_data


def func():
    return multiprocessing.current_process().pid


def grid_search(url):
    X, y, X_train, X_test, y_train, y_test = preprocess_dermatology_data(url)
    pipe_svc = Pipeline([('scl', StandardScaler()),
                         ('clf', SVC(random_state=1))])
    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    param_grid = [{'clf__C': param_range,
                   'clf__kernel': ['linear']},
                  {'clf__C': param_range,
                   'clf__gamma': param_range,
                   'clf__kernel': ['rbf']}]
    gs = GridSearchCV(estimator=pipe_svc,
                      param_grid=param_grid,
                      scoring='accuracy',
                      cv=10,
                      n_jobs=-1)
    gs = gs.fit(X_train, y_train)
    print(gs.best_score_)
    print(gs.best_params_)

    pass


def parallel_func():
    grid_search(url="D:\\mgr\\dermatology\\dermatology.data")
    return Parallel(n_jobs=2)(delayed(func)() for _ in range(2))


if __name__ == '__main__':
    print(Parallel(n_jobs=2)(parallel_func() for _ in range(3)))



