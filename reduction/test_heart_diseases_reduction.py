import unittest
import warnings

from reduction.lda2_reduction import process_lda
from reduction.nmf1_reduction import process_nmf
from reduction.pca3_reduction import process_pca
from reduction.utils import get_run_id


class TestHeartDiseaseDimensionalityReduction(unittest.TestCase):
    def test_process_all_reduction(self):
        warnings.simplefilter("ignore")
        url1 = "D:\\mgr\\heart-disease\\processed.switzerland.data"
        url2 = "D:\\mgr\\heart-disease\\processed.cleveland.data"
        url3 = "D:\\mgr\\heart-disease\\processed.hungarian.data"
        url4 = "D:\\mgr\\heart-disease\\processed.va.data"

        _run_id = get_run_id()

        process_pca(url1, 'Switzerland', n_components=2)
        process_lda(url1, 'Switzerland', n_components=2)
        process_nmf(url1, 'Switzerland', n_components=2)

        process_pca(url2, 'Cleveland', n_components=2)
        process_lda(url2, 'Cleveland', n_components=2)
        process_nmf(url2, 'Cleveland', n_components=2)

        process_pca(url3, 'Hungarian', n_components=2)
        process_lda(url3, 'Hungarian', n_components=2)
        process_nmf(url3, 'Hungarian', n_components=2)

        process_pca(url4, 'Long Beach, CA', n_components=2)
        process_lda(url4, 'Long Beach, CA', n_components=2)
        process_nmf(url4, 'Long Beach, CA', n_components=2)
