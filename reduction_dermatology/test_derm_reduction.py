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

        url1 = "F:\\mgr\\dermatology\\dermatology.data"
        _run_id = get_run_id()

        process_pca(url1, 'Dermatology', n_components=2, run_id=_run_id)
        process_lda(url1, 'Dermatology', n_components=2, run_id=_run_id)
        process_nmf(url1, 'Dermatology', n_components=2, run_id=_run_id)

        process_pca(url1, 'Dermatology', n_components=4, run_id=_run_id)
        process_lda(url1, 'Dermatology', n_components=4, run_id=_run_id)
        process_nmf(url1, 'Dermatology', n_components=4, run_id=_run_id)

        process_without_reduction(url1, 'Dermatology', run_id=_run_id)

    def test_process_all_methods_with_diff_classifiers(self):
        # warnings.filterwarnings("ignore")
        warnings.simplefilter("ignore")
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=DataConversionWarning)

        url1 = "F:\\mgr\\dermatology\\dermatology.data"
        _run_id = get_run_id()

        self._all_reduction(url=url1, n_components=2, run_id=_run_id, classifier='lr')
        self._all_reduction(url=url1, n_components=4, run_id=_run_id, classifier='lr')

        self._all_reduction(url=url1, n_components=2, run_id=_run_id, classifier='svm')
        self._all_reduction(url=url1, n_components=4, run_id=_run_id, classifier='svm')

        self._all_reduction(url=url1, n_components=2, run_id=_run_id, classifier='tree')
        self._all_reduction(url=url1, n_components=4, run_id=_run_id, classifier='tree')

        # process_pca(url1, 'Dermatology', n_components=4, run_id=_run_id)
        # process_lda(url1, 'Dermatology', n_components=4, run_id=_run_id)
        # process_nmf(url1, 'Dermatology', n_components=4, run_id=_run_id)

        process_without_reduction(url1, 'Dermatology', run_id=_run_id, classifier='lr')
        process_without_reduction(url1, 'Dermatology', run_id=_run_id, classifier='svm')
        process_without_reduction(url1, 'Dermatology', run_id=_run_id, classifier='tree')

        print('=========================')
        print('run_id was %s ' % _run_id)
        print('=========================')

    def _all_reduction(self, url, n_components, run_id, classifier):
        process_pca(url, 'Dermatology',
                    n_components=n_components,
                    run_id=run_id,
                    classifier=classifier)
        process_lda(url, 'Dermatology',
                    n_components=n_components,
                    run_id=run_id,
                    classifier=classifier)
        process_nmf(url, 'Dermatology',
                    n_components=n_components,
                    run_id=run_id,
                    classifier=classifier)