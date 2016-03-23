import unittest as test
import numpy as np
import householder as hh
import time

X = np.array(([3],
              [4],
              [0]))
Y = np.array(([0],
              [0],
              [5]))
V = np.array([[7],
              [2],
              [0]])

class TestHouseHolder(test.TestCase):

    def test_prop(self):
        H = hh.householder(X, Y)
        np.testing.assert_array_almost_equal(np.dot(H, X), Y)

    def test_value(self):
        H = hh.householder(X, Y)
        np.testing.assert_array_almost_equal(H, [[0.64, -0.48, 0.6],
                                                 [-0.48, 0.36, 0.8],
                                                 [0.6, 0.8, 0]])



        
        
if __name__ == '__main__':
    test.main()
