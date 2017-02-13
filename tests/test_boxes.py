import unittest
import numpy as np
import sys, os
# add source directory to path
sys.path.append( os.path.abspath( '..' ))
from Boxes import Boxes

class TestPCA( unittest.TestCase ):



    def setUp( self ):
        self.b = Boxes( 25,np.array([[1,2,3],[4,5,6],[7,8,9],[10,0,8],[3,5,-1]]))


    def test_pca_init( self ):
        self.assertEqual( self.b.gridResolution, 25 )
        self.assertRaises(TypeError,Boxes,'1',self.b.data)
        self.assertRaises(ValueError,Boxes,0,self.b.data)
        self.assertRaises(TypeError,Boxes,25,[])
        self.assertRaises(ValueError,Boxes,25,np.array([[1,1,1,1,1,1]]))

    def test_get_min(self):
        minimum=np.array(self.b.get_min())
        np.testing.assert_array_almost_equal(minimum, np.array([1,0,-1]), decimal=6)

    def test_get_max(self):
        maximum=np.array(self.b.get_max())
        np.testing.assert_array_almost_equal(maximum, np.array([10,8,9]), decimal=6)

    def test_rescale(self):
        self.b.rescale_data()
        minimum=np.array(self.b.get_min())
        np.testing.assert_array_almost_equal(minimum, np.array([0,0,0]), decimal=6)
        maximum=np.array(self.b.get_max())
        np.testing.assert_array_almost_equal(maximum, np.array([25,25,25]), decimal=6)

    def test_get_pairs ( self ):
        pass
