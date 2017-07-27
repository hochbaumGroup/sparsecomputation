import numpy as np
import unittest
import sys
import six
import os
from sparsecomputation import SparseComputation, SparseShiftedComputation, SparseHybridComputation
from sparsecomputation.dimreducer import DimReducer, PCA


class TestSparseComputation(unittest.TestCase):

    def setUp(self):
        self.gridResolution = 2
        self.dimReducer = DimReducer(3)
        self.b = SparseComputation(self.dimReducer, self.gridResolution)
        self.data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 0, 8],
                              [3, 5, -1]])

    def test_pca_init(self):
        self.assertEqual(self.b.gridResolution, self.gridResolution)
        self.assertRaises(TypeError, SparseComputation, self.dimReducer, '1')
        self.assertRaises(ValueError, SparseComputation, self.dimReducer, 0)

    def test_rescale(self):
        self.assertRaises(TypeError, self.b._rescale_data, [])
        rescaled_data = self.b._rescale_data(self.data)
        minimum = np.array(np.amin(rescaled_data, axis=0))
        np.testing.assert_array_almost_equal(minimum, np.zeros(3), decimal=6)
        maximum = np.array(np.amax(rescaled_data, axis=0))
        elt = self.gridResolution - 1
        np.testing.assert_array_almost_equal(maximum,
                                             np.array([elt, elt, elt]))
        exp_result = np.array([[0, 0, 0], [0, 1, 1], [1, 1, 1], [1, 0, 1],
                               [0, 1, 0]])
        np.testing.assert_array_almost_equal(rescaled_data,
                                             exp_result)

    def test_get_pairs(self):
        self.assertRaises(TypeError, self.b._get_pairs, [])
        pairs = self.b._get_pairs(self.data)
        expected_pairs = [(1, 2), (0, 4), (0, 1), (0, 3), (0, 2),
                                  (4, 1), (4, 3), (4, 2), (1, 3), (3, 2)]
        six.assertCountEqual(self, pairs, expected_pairs)

    def test_get_pairs_same_box(self):
        pairs = self.b._get_pairs(np.array([[0, 0], [0, 0]]))
        expected_pairs = [(0, 1)]
        six.assertCountEqual(self, pairs, expected_pairs)

class TestSparseShiftedComputation(unittest.TestCase):

    def setUp(self):
        self.gridResolution = 3
        self.dimReducer = PCA(3)
        self.sc = SparseComputation(self.dimReducer, self.gridResolution*2)
        self.ssc = SparseShiftedComputation(self.dimReducer, self.gridResolution)
        self.data = np.array([[3, 5, 2, 8, 4, 4, 2, -10, 8, -2],
                        [8, 2, -6, 9, 5, 6, 6, 2, 9, -7],
                        [1, -1, -6, 0, 1, -7, -6, 6, 0, 2],
                        [8, 5, 8, 1, 1, 7, 2, 4, -7, -9],
                        [-10, 10, 2, 5, -7, -10, -6, -3, 7, -4],
                        [-6, -10, -8, 2, 2, 0, 4, 7, 9, -4],
                        [-6, 9, 5, 4, 6, -3, 8, -10, 4, 2],
                        [6, -8, -5, 4, -4, -1, -2, 5, 8, 6],
                        [10, -2, 8, 9, -4, -8, 7, 10, -7, 0],
                        [10, 4, 4, -9, -9, 8, -6, 1, -7, 10]])

    def test_ssc_output(self):
        pairs_sc = self.sc.get_similar_indices(self.data)
        pairs_ssc = self.ssc.get_similar_indices(self.data)
        self.assertEqual(len(pairs_sc), len(pairs_ssc))

class TestSparseHybridComputation(unittest.TestCase):

    def setUp(self):
        self.gridResolution = 3
        self.dimReducer = PCA(3)
        self.sc = SparseComputation(self.dimReducer, self.gridResolution*2)
        self.shc = SparseHybridComputation(self.dimReducer, self.gridResolution)
        self.ssc = SparseShiftedComputation(self.dimReducer, self.gridResolution)
        self.data = np.array([[3, 5, 2, 8, 4, 4, 2, -10, 8, -2],
                        [8, 2, -6, 9, 5, 6, 6, 2, 9, -7],
                        [1, -1, -6, 0, 1, -7, -6, 6, 0, 2],
                        [8, 5, 8, 1, 1, 7, 2, 4, -7, -9],
                        [-10, 10, 2, 5, -7, -10, -6, -3, 7, -4],
                        [-6, -10, -8, 2, 2, 0, 4, 7, 9, -4],
                        [-6, 9, 5, 4, 6, -3, 8, -10, 4, 2],
                        [6, -8, -5, 4, -4, -1, -2, 5, 8, 6],
                        [10, -2, 8, 9, -4, -8, 7, 10, -7, 0],
                        [10, 4, 4, -9, -9, 8, -6, 1, -7, 10]])

    def test_shc_output(self):
        pairs_sc = self.sc.get_similar_indices(self.data)
        pairs_ssc = self.ssc.get_similar_indices(self.data)
        pairs_shc = self.shc.get_similar_indices(self.data)
        self.assertEqual(len(pairs_sc), len(pairs_shc))
        self.assertEqual(len(pairs_ssc), len(pairs_shc))