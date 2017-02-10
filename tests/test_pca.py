import unittest
import numpy as np
import sys, os
# add source directory to path
sys.path.append( os.path.abspath( '..' ))
from PCA import PCA

class TestPCA( unittest.TestCase ):

    def setUp( self ):
        self.p = PCA( 3)

    def test_pca_init( self ):
        self.assertEqual( self.p.dimLow, 3 )

    def test_input_exceptions( self ):
        data=[[1,2]]
        self.assertRaises(TypeError,self.p.fit_transform,data)
        data=np.array(data)
        self.assertRaises(ValueError,self.p.fit_transform,data)
