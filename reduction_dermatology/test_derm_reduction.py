import unittest
import warnings

from sklearn.exceptions import DataConversionWarning

from reduction.utils import get_run_id
from reduction_dermatology.derm_lda1_reduction import process_lda
from reduction_dermatology.derm_nmf1_reduction import process_nmf
from reduction_dermatology.derm_pca3_reduction import process_pca
from reduction_dermatology.derm_without_reduction import process_without_reduction


class TestDermatologyReduction(unittest.TestCase):
    def test_process_all_methods(self):
        # warnings.filterwarnings("ignore")
        warnings.simplefilter("ignore")
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=DataConversionWarning)

        url1 = "D:\\mgr\\dermatology\\dermatology.data"
        _run_id = get_run_id()

        process_pca(url1, 'Dermatology', n_components=2, run_id=_run_id)
        process_lda(url1, 'Dermatology', n_components=2, run_id=_run_id)
        process_nmf(url1, 'Dermatology', n_components=2, run_id=_run_id)

        process_pca(url1, 'Dermatology', n_components=4, run_id=_run_id)
        process_lda(url1, 'Dermatology', n_components=4, run_id=_run_id)
        process_nmf(url1, 'Dermatology', n_components=4, run_id=_run_id)

        process_without_reduction(url1, 'Dermatology', run_id=_run_id)
