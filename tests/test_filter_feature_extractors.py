from src.preprocessing.feature_extraction.text.filtering import TermStrengthFeatureExtractor
import pytest
import numpy as np 
from scipy.sparse import csr_matrix

@pytest.fixture
def count_matrix():
    arr = np.array([[0,1,0,0,0,1,0], [1,1,1,0,0,0,0], [1,1,0,0,0,1,1],  [1,1,0,1,0,1,0],  [1,1,0,1,0,1,1]])
    return csr_matrix(arr)

class TestTermStrengthFeatureExtractor:
    def test_fit(self, count_matrix):
        extractor = TermStrengthFeatureExtractor()
        extractor.fit(count_matrix, np.zeros(5))
        
        expected = np.array([0.6,1,0,1/7,0,0.6,1/7])
        s_t = np.squeeze(extractor.term_strength[0])

        np.testing.assert_almost_equal(s_t, expected)
    
    def test_transform(self, count_matrix):
        extractor = TermStrengthFeatureExtractor()
        extractor.fit(count_matrix, np.zeros(5))

        expected = np.array([
            [0, 0.6, 0.6, 0.6, 0.6],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0 , 0],
            [0, 0, 0, 1/7, 1/7],
            [0, 0, 0, 0 , 0],
            [0.6, 0, 0.6, 0.6, 0.6],
            [0, 0, 1/7, 0, 1/7]
        ]).T

        transformed = extractor.transform(count_matrix)[0].toarray()

        np.testing.assert_almost_equal(transformed, expected)

    def test_pruning(self, count_matrix):
        extractor = TermStrengthFeatureExtractor()
        extractor.fit(count_matrix, np.zeros(5))

        expected = np.array([0,1,5])

        keep_idx = extractor.prune(n_std=2)[0]

        np.testing.assert_equal(keep_idx, expected)