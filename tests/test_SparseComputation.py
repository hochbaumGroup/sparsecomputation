import numpy as np
import unittest
import sys
import os
from SparseComputation import SparseComputation
from DimReducer import DimReducer
# add source directory to path
sys.path.append(os.path.abspath('..'))


class TestPCA(unittest.TestCase):

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

    def test_get_min(self):
        self.assertRaises(TypeError, self.b._get_min, [])
        self.assertRaises(ValueError, self.b._get_min, np.array([np.ones(6)]))
        minimum = np.array(self.b._get_min(self.data))
        np.testing.assert_array_almost_equal(minimum, np.array([1, 0, -1]))

    def test_get_max(self):
        self.assertRaises(TypeError, self.b._get_max, [])
        self.assertRaises(ValueError, self.b._get_max, np.array([np.ones(6)]))
        maximum = np.array(self.b._get_max(self.data))
        np.testing.assert_array_almost_equal(maximum, np.array([10, 8, 9]))

    def test_rescale(self):
        self.assertRaises(TypeError, self.b._rescale_data, [])
        self.assertRaises(ValueError, self.b._rescale_data,
                          np.array([np.ones(6)]))
        rescaled_data = self.b._rescale_data(self.data)
        minimum = np.array(self.b._get_min(rescaled_data))
        np.testing.assert_array_almost_equal(minimum, np.zeros(3), decimal=6)
        maximum = np.array(self.b._get_max(rescaled_data))
        elt = self.gridResolution - 1
        np.testing.assert_array_almost_equal(maximum,
                                             np.array([elt, elt, elt]))

    def test_get_pairs(self):
        self.assertRaises(TypeError, self.b.get_sim_indices, [])
        self.assertRaises(ValueError, self.b.get_sim_indices,
                          np.array([np.ones(6)]))
        pairs = np.array(self.b.get_sim_indices(self.data))
        expected_pairs = np.array([(0, 0), (0, 1), (0, 4), (1, 0), (1, 1),
                                  (1, 4), (4, 0), (4, 1), (4, 4), (0, 3),
                                  (1, 3), (4, 3), (3, 3), (3, 0), (3, 1),
                                  (3, 4), (2, 2)])
        np.testing.assert_array_almost_equal(pairs, expected_pairs, decimal=6)
