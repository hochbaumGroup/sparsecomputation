import unittest
import numpy as np
import sys
import os
from SparseComputation import PCA


class TestPCA(unittest.TestCase):

    def setUp(self):
        self.p = PCA(3)

    def test_pca_init(self):
        self.assertEqual(self.p.dimLow, 3)
        self.assertRaises(TypeError, PCA, 'a')
        self.assertRaises(ValueError, PCA, -1)

    def test_input_exceptions(self):
        data = [[1, 2]]
        self.assertRaises(TypeError, self.p.fit_transform, data)
        data = np.array(data)
        self.assertRaises(ValueError, self.p.fit_transform, data)

    def test_pca(self):
        data = np.array([[3, 5, 2, 8, 4, 4, 2, -10, 8, -2],
                        [8, 2, -6, 9, 5, 6, 6, 2, 9, -7],
                        [1, -1, -6, 0, 1, -7, -6, 6, 0, 2],
                        [8, 5, 8, 1, 1, 7, 2, 4, -7, -9],
                        [-10, 10, 2, 5, -7, -10, -6, -3, 7, -4],
                        [-6, -10, -8, 2, 2, 0, 4, 7, 9, -4],
                        [-6, 9, 5, 4, 6, -3, 8, -10, 4, 2],
                        [6, -8, -5, 4, -4, -1, -2, 5, 8, 6],
                        [10, -2, 8, 9, -4, -8, 7, 10, -7, 0],
                        [10, 4, 4, -9, -9, 8, -6, 1, -7, 10]])
        result = np.array([[-8.71599884143, -7.78390390247, -5.66048721019],
                          [-5.88506238249, 4.33602496305, -13.3854986002],
                          [1.90004430443, 7.30110993281, 8.25058818896],
                          [9.82812379318, -7.82961491623, -9.49534227851],
                          [-11.2055273148, -6.71449401142, 14.317169179],
                          [-8.19523837297, 15.4248893386, -0.470369156087],
                          [-11.6697682118, -12.8178482155, 0.215189474438],
                          [2.42284114633, 12.6009577226, 3.22559475278],
                          [11.235644912, 1.82991407312, -4.40780556368],
                          [20.2849409675, -6.34703498457, 7.41096121351]])
        newdata = self.p.fit_transform(data)
        np.testing.assert_array_almost_equal(result, newdata, decimal=6)
