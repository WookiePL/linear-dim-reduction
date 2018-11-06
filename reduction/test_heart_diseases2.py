import time

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

        url1 = "F:\\mgr\\heart-disease\\processed.switzerland.data"
        url2 = "F:\\mgr\\heart-disease\\processed.cleveland.data"
        url3 = "F:\\mgr\\heart-disease\\processed.hungarian.data"
        url4 = "F:\\mgr\\heart-disease\\processed.va.data"

        _run_id = get_run_id()

        #cls._all_cleveland(url2, 'Cleveland', _run_id)
        #cls._all_switzerland(url1, 'Switzerland', _run_id)
        cls._all_hungarian(url2, 'Hungarian', _run_id)
        # cls._all_long_beach(url2, 'Long Beach, CA', _run_id)



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


        print('=========================')
        print('run_id was %s ' % _run_id)
        print('=========================')

    @classmethod
    def _all_cleveland(cls, url, title, run_id):
        process_pca(url, title, n_components=2, run_id=run_id, classifier='lr')
        time.sleep(1)
        process_pca(url, title, n_components=5, run_id=run_id, classifier='lr')
        time.sleep(1)
        process_pca(url, title, n_components=2, run_id=run_id, classifier='svm')
        time.sleep(1)
        process_pca(url, title, n_components=5, run_id=run_id, classifier='svm')
        time.sleep(1)
        process_pca(url, title, n_components=2, run_id=run_id, classifier='tree')
        time.sleep(1)
        process_pca(url, title, n_components=5, run_id=run_id, classifier='tree')
        time.sleep(1)
        process_lda(url, title, n_components=1, run_id=run_id, classifier='lr')
        time.sleep(1)
        process_lda(url, title, n_components=1, run_id=run_id, classifier='svm')
        time.sleep(1)
        process_lda(url, title, n_components=1, run_id=run_id, classifier='tree')
        time.sleep(1)
        process_nmf(url, title, n_components=2, run_id=run_id, classifier='lr')
        time.sleep(1)
        process_nmf(url, title, n_components=5, run_id=run_id, classifier='lr')
        time.sleep(1)
        process_nmf(url, title, n_components=2, run_id=run_id, classifier='svm')
        time.sleep(1)
        process_nmf(url, title, n_components=5, run_id=run_id, classifier='svm')
        time.sleep(1)
        process_nmf(url, title, n_components=2, run_id=run_id, classifier='tree')
        time.sleep(1)
        process_nmf(url, title, n_components=5, run_id=run_id, classifier='tree')
        time.sleep(1)
        process_without_reduction(url, title, run_id=run_id, classifier='lr')
        time.sleep(1)
        process_without_reduction(url, title, run_id=run_id, classifier='svm')
        time.sleep(1)
        process_without_reduction(url, title, run_id=run_id, classifier='tree')
        time.sleep(1)

    @classmethod
    def _all_switzerland(cls, url, title, run_id):
        process_pca(url, title, n_components=3, run_id=run_id, classifier='lr')
        time.sleep(1)
        process_pca(url, title, n_components=6, run_id=run_id, classifier='lr')
        time.sleep(1)
        process_pca(url, title, n_components=3, run_id=run_id, classifier='svm')
        time.sleep(1)
        process_pca(url, title, n_components=6, run_id=run_id, classifier='svm')
        time.sleep(1)
        process_pca(url, title, n_components=3, run_id=run_id, classifier='tree')
        time.sleep(1)
        process_pca(url, title, n_components=6, run_id=run_id, classifier='tree')
        time.sleep(1)
        process_lda(url, title, n_components=1, run_id=run_id, classifier='lr')
        time.sleep(1)
        process_lda(url, title, n_components=1, run_id=run_id, classifier='svm')
        time.sleep(1)
        process_lda(url, title, n_components=1, run_id=run_id, classifier='tree')
        time.sleep(1)
        process_nmf(url, title, n_components=3, run_id=run_id, classifier='lr')
        time.sleep(1)
        process_nmf(url, title, n_components=6, run_id=run_id, classifier='lr')
        time.sleep(1)
        process_nmf(url, title, n_components=3, run_id=run_id, classifier='svm')
        time.sleep(1)
        process_nmf(url, title, n_components=6, run_id=run_id, classifier='svm')
        time.sleep(1)
        process_nmf(url, title, n_components=3, run_id=run_id, classifier='tree')
        time.sleep(1)
        process_nmf(url, title, n_components=6, run_id=run_id, classifier='tree')
        time.sleep(1)
        process_without_reduction(url, title, run_id=run_id, classifier='lr')
        time.sleep(1)
        process_without_reduction(url, title, run_id=run_id, classifier='svm')
        time.sleep(1)
        process_without_reduction(url, title, run_id=run_id, classifier='tree')
        time.sleep(1)

    @classmethod
    def _all_hungarian(cls, url, title, run_id):
        process_pca(url, title, n_components=2, run_id=run_id, classifier='lr')
        time.sleep(1)
        process_pca(url, title, n_components=3, run_id=run_id, classifier='lr')
        time.sleep(1)
        process_pca(url, title, n_components=5, run_id=run_id, classifier='lr')
        time.sleep(1)
        
        process_pca(url, title, n_components=2, run_id=run_id, classifier='svm')
        time.sleep(1)
        process_pca(url, title, n_components=3, run_id=run_id, classifier='svm')
        time.sleep(1)
        process_pca(url, title, n_components=5, run_id=run_id, classifier='svm')
        time.sleep(1)
        
        process_pca(url, title, n_components=2, run_id=run_id, classifier='tree')
        time.sleep(1)
        process_pca(url, title, n_components=3, run_id=run_id, classifier='tree')
        time.sleep(1)
        process_pca(url, title, n_components=5, run_id=run_id, classifier='tree')
        time.sleep(1)
        
        
        
        process_lda(url, title, n_components=1, run_id=run_id, classifier='lr')
        time.sleep(1)
        process_lda(url, title, n_components=1, run_id=run_id, classifier='svm')
        time.sleep(1)
        process_lda(url, title, n_components=1, run_id=run_id, classifier='tree')
        time.sleep(1)


        process_nmf(url, title, n_components=2, run_id=run_id, classifier='lr')
        time.sleep(1)
        process_nmf(url, title, n_components=3, run_id=run_id, classifier='lr')
        time.sleep(1)
        process_nmf(url, title, n_components=5, run_id=run_id, classifier='lr')
        time.sleep(1)
    
        process_nmf(url, title, n_components=2, run_id=run_id, classifier='svm')
        time.sleep(1)
        process_nmf(url, title, n_components=3, run_id=run_id, classifier='svm')
        time.sleep(1)
        process_nmf(url, title, n_components=5, run_id=run_id, classifier='svm')
        time.sleep(1)

        process_nmf(url, title, n_components=2, run_id=run_id, classifier='tree')
        time.sleep(1)
        process_nmf(url, title, n_components=3, run_id=run_id, classifier='tree')
        time.sleep(1)
        process_nmf(url, title, n_components=5, run_id=run_id, classifier='tree')
        time.sleep(1)



        process_without_reduction(url, title, run_id=run_id, classifier='lr')
        time.sleep(1)
        process_without_reduction(url, title, run_id=run_id, classifier='svm')
        time.sleep(1)
        process_without_reduction(url, title, run_id=run_id, classifier='tree')
        time.sleep(1)


test = TestHeartDiseaseDimensionalityReduction
test.test_process_all_reduction()
