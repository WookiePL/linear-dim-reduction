import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA

# first variance charts method
def show_intrinsic_dimensions(url, title):
    # załadowanie zbioru danych do Pandas DataFrame
    df = pd.read_csv(url,
                     names=['erythema',
                            'scaling',
                            'definite borders',
                            'itching',
                            'koebner phenomenon',
                            'polygonal papules',
                            'follicular papules',
                            'oral mucosal involvement',
                            'knee and elbow involvement',
                            'scalp involvement',
                            'family history, (0 or 1)',
                            'melanin incontinence',
                            'eosinophils in the infiltrate',
                            'PNL infiltrate',
                            'fibrosis of the papillary dermis',
                            'exocytosis',
                            'acanthosis',
                            'hyperkeratosis',
                            'parakeratosis',
                            'clubbing of the rete ridges',
                            'elongation of the rete ridges',
                            'thinning of the suprapapillary epidermis',
                            'spongiform pustule',
                            'munro microabcess',
                            'focal hypergranulosis',
                            'disappearance of the granular layer',
                            'vacuolisation and damage of basal layer',
                            'spongiosis',
                            'saw-tooth appearance of retes',
                            'follicular horn plug',
                            'perifollicular parakeratosis',
                            'inflammatory monoluclear inflitrate',
                            'band-like infiltrate',
                            'Age (linear)'])

    df = df.replace('?', '0')
    df.convert_objects(convert_numeric=True)

    # normalize data
    data_scaled = pd.DataFrame(preprocessing.scale(df), columns=df.columns)

    pca = PCA()
    pca.fit(df)

    features2 = range(pca.n_components_)

    plt.bar(features2, pca.explained_variance_)
    plt.xticks(features2)
    plt.ylabel('variance')
    plt.xlabel('PCA feature')
    plt.show()

    indexes = []
    for i in range(34):
        indexes.append('PC-' + (1 + i).__str__())

    pd.set_option('display.max_columns', None)
    print(pd.DataFrame(pca.components_, columns=data_scaled.columns, index=indexes))


url1 = "F:\\mgr\\dermatology\\dermatology.data"

show_intrinsic_dimensions(url1, 'Dermatology')
