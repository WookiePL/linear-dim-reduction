import pandas as pd
from IPython.display import HTML

from reduction.utils import preprocess_heart_disease_data


def preprocess(url, name):
    X, y, X_train, X_test, y_train, y_test = preprocess_heart_disease_data(url)
    df = pd.DataFrame(X_train)
    df.head()
    h = HTML(df.to_html())
    my_file = open('some_file.html', 'w')
    my_file.write(h.data)
    my_file.close()


url1 = "D:\\mgr\\heart-disease\\processed.switzerland.data"
url2 = "D:\\mgr\\heart-disease\\processed.cleveland.data"
url3 = "D:\\mgr\\heart-disease\\processed.hungarian.data"
url4 = "D:\\mgr\\heart-disease\\processed.va.data"

# preprocess(url1, 'Switzerland')
preprocess(url2, 'Cleveland')
# preprocess(url3, 'Hungarian')
# preprocess(url4, 'Long Beach, CA')
