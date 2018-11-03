import unittest
import warnings

from sklearn.exceptions import DataConversionWarning

from reduction_dermatology.derm_lda1_reduction import process_lda
from reduction_dermatology.derm_nmf1_reduction import process_nmf
from reduction_dermatology.derm_pca3_reduction import process_pca
from reduction_dermatology.derm_without_reduction import process_without_reduction


class TestDermatologyReduction(unittest.TestCase):
    def test_process_pca(self):
        # warnings.filterwarnings("ignore")
        warnings.simplefilter("ignore")
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=DataConversionWarning)

        url1 = "D:\\mgr\\dermatology\\dermatology.data"

        process_pca(url1, 'Dermatology', n_components=2)
        process_lda(url1, 'Dermatology', n_components=2)
        process_nmf(url1, 'Dermatology', n_components=2)

        process_pca(url1, 'Dermatology', n_components=4)
        process_lda(url1, 'Dermatology', n_components=4)
        process_nmf(url1, 'Dermatology', n_components=4)

        process_without_reduction(url1, 'Dermatology')
