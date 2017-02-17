import numpy as np
import unittest
import sys, os
# add source directory to path
sys.path.append( os.path.abspath( '..' ))
from Boxes import Boxes

class TestPCA( unittest.TestCase ):



    def setUp( self ):
        self.gridResolution=2
        self.b = Boxes( self.gridResolution )
        self.data=np.array([[1,2,3],[4,5,6],[7,8,9],[10,0,8],[3,5,-1]])

    def test_pca_init( self ):
        self.assertEqual( self.b.gridResolution, self.gridResolution )
        self.assertRaises(TypeError,Boxes,'1')
        self.assertRaises(ValueError,Boxes,0)

    def test_get_min(self):
        self.assertRaises(TypeError,self.b.get_min,[])
        self.assertRaises(ValueError,self.b.get_min,np.array([[1,1,1,1,1,1]]))
        minimum=np.array(self.b.get_min(self.data))
        np.testing.assert_array_almost_equal(minimum, np.array([1,0,-1]), decimal=6)

    def test_get_max(self):
        self.assertRaises(TypeError,self.b.get_max,[])
        self.assertRaises(ValueError,self.b.get_max,np.array([[1,1,1,1,1,1]]))
        maximum=np.array(self.b.get_max(self.data))
        np.testing.assert_array_almost_equal(maximum, np.array([10,8,9]), decimal=6)

    def test_rescale(self):
        self.assertRaises(TypeError,self.b.rescale_data,[])
        self.assertRaises(ValueError,self.b.rescale_data,np.array([[1,1,1,1,1,1]]))
        rescaled_data=self.b.rescale_data(self.data)
        minimum=np.array(self.b.get_min(rescaled_data))
        np.testing.assert_array_almost_equal(minimum, np.array([0,0,0]), decimal=6)
        maximum=np.array(self.b.get_max(rescaled_data))
        np.testing.assert_array_almost_equal(maximum, np.array([self.gridResolution,self.gridResolution,self.gridResolution]), decimal=6)

    def test_get_pairs ( self ):
        self.assertRaises(TypeError,self.b.get_pairs,[])
        self.assertRaises(ValueError,self.b.get_pairs,np.array([[1,1,1,1,1,1]]))
        pairs=np.array(self.b.get_pairs(self.data))
        expected_pairs=np.array([(0,1),(0,4),(1,2),(1,4)])
        np.testing.assert_array_almost_equal(pairs,expected_pairs,decimal=6)
