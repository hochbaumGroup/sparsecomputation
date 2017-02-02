import unittest
import sys, os
# add source directory to path
sys.path.append( os.path.abspath( '..' ))
from ApproximatePCA import ApproximatePCA

class TestApproximatePCA( unittest.TestCase ):

    def setUp( self ):
        self.ap = ApproximatePCA( 3, 0.01, 1, 100, 150 )

    def test_approx_pca_init( self ):
        self.assertEqual( self.ap.dimLow, 3 )
        self.assertEqual( self.ap.percRow, 0.01 )
        self.assertEqual( self.ap.percCol, 1 )
        self.assertEqual( self.ap.minRow, 100 )
        self.assertEqual( self.ap.minCol, 150 )

    def test_approx_pca_fit_transform_small_dim( self ):
        pass

    def test_approx_pca_fit_transform_full_dim( self ):
        pass

    def test_approx_pca_fit_transform_low_min_row( self ):
        pass

    def test_approx_pca_fit_transform_low_min_col( self ):
        pass
