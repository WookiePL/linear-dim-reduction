import unittest
import warnings

from sklearn.exceptions import DataConversionWarning

from reduction_dermatology.derm_lda1_reduction import process_lda
from reduction_dermatology.derm_nmf1_reduction import process_nmf
from reduction_dermatology.derm_pca3_reduction import process_pca


class TestDermatologyPCA(unittest.TestCase):
    def test_process_pca(self):
        # warnings.filterwarnings("ignore")
        warnings.simplefilter("ignore")
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=DataConversionWarning)

        url1 = "D:\\mgr\\dermatology\\dermatology.data"

        process_pca(url1, 'Dermatology', 2)
        process_lda(url1, 'Dermatology')
        process_nmf(url1, 'Dermatology')
