import unittest
import sys
import os
import numpy as np
from sparsecomputation.dimreducer import DimReducer


class TestDimReducer(unittest.TestCase):
    def setUp(self):
        self.dr = DimReducer(3)

    def test_dim_reduce_init(self):
        self.assertEqual(self.dr.dimLow, 3)
