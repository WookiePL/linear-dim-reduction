import unittest
import warnings

from reduction.lda2_reduction import process_lda
from reduction.nmf1_reduction import process_nmf
from reduction.pca3_reduction import process_pca
from reduction.utils import get_run_id
from reduction.without_reduction import process_without_reduction


class TestHeartDiseaseDimensionalityReduction:
    def __init__(self):
        pass
    @classmethod
    def test_process_all_reduction(cls):
        warnings.simplefilter("ignore")
        url1 = "D:\\mgr\\heart-disease\\processed.switzerland.data"
        url2 = "D:\\mgr\\heart-disease\\processed.cleveland.data"
        url3 = "D:\\mgr\\heart-disease\\processed.hungarian.data"
        url4 = "D:\\mgr\\heart-disease\\processed.va.data"

        _run_id = get_run_id()

        cls._all_cleveland_pca(url2, _run_id)
        # self._all_cleveland_lda(url2, _run_id)
        # self._all_cleveland_nmf(url2, _run_id)


        # process_lda(url2, 'Cleveland', n_components=2, run_id=_run_id)
        # process_nmf(url2, 'Cleveland', n_components=2, run_id=_run_id)
        #
        # process_pca(url1, 'Switzerland', n_components=2, run_id=_run_id)
        # process_lda(url1, 'Switzerland', n_components=2, run_id=_run_id)
        # process_nmf(url1, 'Switzerland', n_components=2, run_id=_run_id)
        #
        #
        # process_pca(url3, 'Hungarian', n_components=2, run_id=_run_id)
        # process_lda(url3, 'Hungarian', n_components=2, run_id=_run_id)
        # process_nmf(url3, 'Hungarian', n_components=2, run_id=_run_id)
        #
        # process_pca(url4, 'Long Beach, CA', n_components=2, run_id=_run_id)
        # process_lda(url4, 'Long Beach, CA', n_components=2, run_id=_run_id)
        # process_nmf(url4, 'Long Beach, CA', n_components=2, run_id=_run_id)

    @classmethod
    def _all_cleveland_pca(url, run_id):
        process_pca(url, 'Cleveland', n_components=2, run_id=run_id, classifier='lr')
        process_pca(url, 'Cleveland', n_components=5, run_id=run_id, classifier='lr')

        process_pca(url, 'Cleveland', n_components=2, run_id=run_id, classifier='svm')
        process_pca(url, 'Cleveland', n_components=5, run_id=run_id, classifier='svm')

        process_pca(url, 'Cleveland', n_components=2, run_id=run_id, classifier='tree')
        process_pca(url, 'Cleveland', n_components=5, run_id=run_id, classifier='tree')


test = TestHeartDiseaseDimensionalityReduction
test.test_process_all_reduction()

    #
    # def test_process_all_methods_with_diff_classifiers(self):
    #     # warnings.filterwarnings("ignore")
    #     warnings.simplefilter("ignore")
    #
    #     url1 = "D:\\mgr\\heart-disease\\processed.switzerland.data"
    #     url2 = "D:\\mgr\\heart-disease\\processed.cleveland.data"
    #     url3 = "D:\\mgr\\heart-disease\\processed.hungarian.data"
    #     url4 = "D:\\mgr\\heart-disease\\processed.va.data"
    #     _run_id = get_run_id()
    #
    #     self._all_for_dataset(url2, 'Cleveland', n_components=2, run_id=_run_id)
    #     self._all_for_dataset(url1, 'Switzerland', n_components=2, run_id=_run_id)
    #     self._all_for_dataset(url3, 'Hungarian', n_components=2, run_id=_run_id)
    #     self._all_for_dataset(url4, 'Long Beach, CA', n_components=2, run_id=_run_id)
    #
    #     self._all_for_dataset(url2, 'Cleveland', n_components=4, run_id=_run_id)
    #     self._all_for_dataset(url1, 'Switzerland', n_components=4, run_id=_run_id)
    #     self._all_for_dataset(url3, 'Hungarian', n_components=4, run_id=_run_id)
    #     self._all_for_dataset(url4, 'Long Beach, CA', n_components=4, run_id=_run_id)
    #
    #     print('=========================')
    #     print('run_id was %s ' % _run_id)
    #     print('=========================')
    #
    # def _all_for_dataset(self, dataset_url, dataset_title, n_components, run_id):
    #     self._all_reduction(dataset_url, dataset_title, n_components, run_id, 'lr')
    #     self._all_reduction(dataset_url, dataset_title, n_components, run_id, 'svm')
    #     self._all_reduction(dataset_url, dataset_title, n_components, run_id, 'tree')
    #     self._all_without_reduction(dataset_url, dataset_title, run_id, 'lr')
    #     self._all_without_reduction(dataset_url, dataset_title, run_id, 'svm')
    #     self._all_without_reduction(dataset_url, dataset_title, run_id, 'tree')
    #
    # def _all_reduction(self, url, title, n_components, run_id, classifier):
    #     process_pca(url, title,
    #                 n_components=n_components,
    #                 run_id=run_id,
    #                 classifier=classifier)
    #     process_lda(url, title,
    #                 n_components=n_components,
    #                 run_id=run_id,
    #                 classifier=classifier)
    #     # process_nmf(url, title,
    #     #             n_components=n_components,
    #     #             run_id=run_id,
    #     #             classifier=classifier)
    #
    # def _all_without_reduction(self, url, title, run_id, classifier):
    #     process_without_reduction(url, title,
    #                               run_id=run_id,
    #                               classifier=classifier)
