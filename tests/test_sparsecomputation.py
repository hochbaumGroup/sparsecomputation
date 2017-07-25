import numpy as np
import unittest
import sys
import six
import os
from sparsecomputation import SparseComputation, SparseShiftedComputation
from sparsecomputation.dimreducer import DimReducer


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
        output = self.b._get_pairs(self.data)
        expected_pairs = [(1, 2), (0, 4), (0, 1), (0, 3), (0, 2),
                                  (4, 1), (4, 3), (4, 2), (1, 3), (3, 2)]
        six.assertCountEqual(self, output['pairs'], expected_pairs)

    def test_get_pairs_same_box(self):
        output = self.b._get_pairs(np.array([[0, 0], [0, 0]]))
        expected_pairs = [(0, 1)]
        six.assertCountEqual(self, output['pairs'], expected_pairs)

class TestSparseShiftedComputation(unittest.TestCase):

    def setUp(self):
        self.gridResolution = 3
        self.dimReducer = DimReducer(2)
        self.b = SparseShiftedComputation(self.dimReducer, self.gridResolution)
        self.data = np.array([[-1, -5], [-0.5,-4.5], [-0.5, -3.5], [0,-3],
                              [-0.5,-2.5], [1.5,-4.5], [1.5,-2.5], [2,-2]])
    def test_ssc_init(self):
        self.assertEqual(self.b.gridResolution, self.gridResolution)
        self.assertRaises(TypeError, SparseShiftedComputation,
                          self.dimReducer, '1')
        self.assertRaises(ValueError, SparseShiftedComputation,
                          self.dimReducer, 0)
