import numpy as np
import unittest
import sys
import six
import os
from SparseComputation.SparseReduce import SparseReduce
from SparseComputation.DimReducer import PCA


class TestSparseReduce(unittest.TestCase):

    def setUp(self):
        self.gridResolution = 2
        self.subblockResolution = 3
        self.dimReducer = PCA(3)
        self.sr = SparseReduce(self.dimReducer, self.gridResolution,
                               self.subblockResolution)
        self.data = np.array([[-1, -12, 5], [-1, -12, 6], [-1, -11, 6],
                              [5, 0, 23], [0, -10, 8]])

    def test_rescale(self):
        self.assertRaises(TypeError, self.sr._rescale_data, [])
        rescaled_data = self.sr._rescale_data(self.data)
        minimum = np.array(np.amin(rescaled_data, axis=0))
        np.testing.assert_array_almost_equal(minimum, np.zeros(3), decimal=6)
        maximum = np.array(np.amax(rescaled_data, axis=0))
        elt = self.gridResolution*self.subblockResolution - 1
        np.testing.assert_array_almost_equal(maximum,
                                             np.array([elt, elt, elt]))
        exp_result = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [5, 5, 5],
                               [1, 1, 1]])
        np.testing.assert_array_almost_equal(rescaled_data,
                                             exp_result)

    def test_get_subblocks(self):
        rescaled_data = self.sr._rescale_data(self.data)
        subblocks = self.sr._get_sub_blocks(rescaled_data)
        expected = {}
        expected[(0, 0, 0)] = [0, 1, 2]
        expected[(5, 5, 5)] = [3]
        expected[(1, 1, 1)] = [4]
        np.testing.assert_array_almost_equal(expected.keys(), subblocks.keys())
        for key in subblocks.keys():
            np.testing.assert_array_almost_equal(expected[key], subblocks[key])

    def test_get_boxes(self):
        rescaled_data = self.sr._rescale_data(self.data)
        subblocks = self.sr._get_sub_blocks(rescaled_data)
        blocks = self.sr._get_boxes(subblocks, rescaled_data)
        expected = {}
        expected[(0, 0, 0)] = [(1, 1, 1), (0, 0, 0)]
        expected[(1, 1, 1)] = [(5, 5, 5)]
        np.testing.assert_array_almost_equal(expected.keys(), blocks.keys())
        for key in blocks.keys():
            np.testing.assert_array_almost_equal(expected[key], blocks[key])

    def test_get_reduce(self):
        rescaled_data = self.sr._rescale_data(self.data)
        subblocks = self.sr._get_sub_blocks(rescaled_data)
        blocks = self.sr._get_boxes(subblocks, rescaled_data)
        pairs = self.sr._get_pairs(rescaled_data, blocks)
        exp_pairs = [((0, 0, 0), (1, 1, 1)), ((0, 0, 0), (5, 5, 5)),
                     ((1, 1, 1), (5, 5, 5))]
        print pairs
        print self.sr.get_Reduced_data(self.data)
        print self.sr.sparseReduceComputation(self.data,label = np.array([0,1,1,1,1]))
        print self.sr.sparseReduceComputation(self.data)
