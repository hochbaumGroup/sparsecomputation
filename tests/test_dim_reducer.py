import unittest
import sys
import os
from DimReducer import DimReducer
# add source directory to path
sys.path.append(os.path.abspath('..'))


class TestDimReducer(unittest.TestCase):
    def setUp(self):
        self.dr = DimReducer(3)

    def test_dim_reduce_init(self):
        self.assertEqual(self.dr.dimLow, 3)
