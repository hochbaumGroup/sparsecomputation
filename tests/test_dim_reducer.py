import unittest
import sys, os
# add source directory to path
sys.path.append( os.path.abspath( '..' ))
from DimReducer import DimReducer

class TestDimReducer(unittest.TestCase ):
    def setUp(self):
        self.dr = DimReducer( 3 )

    def test_dim_reduce_init( self ):
        self.assertEqual( self.dr.dimLow, 3 )
