import unittest as test
import numpy as np
import exo2 as bd


A = np.array([[1, 2, 3, 4],
              [7, 3, 9, 2],
              [3, 0, 4, 5]])

B = np.array([[1, 1, 0, 0],
              [0, 1, 1, 0],
              [0, 0, 1, 1],
              [0, 0, 0, 1]])

C = np.array([[1, 1, 0, 1],
              [0, 1, 1, 0],
              [0, 0, 1, 1],
              [0, 0, 0, 1]])


def is_bidiag(mat):
    n, m = np.shape(mat)
    for i in range(0, n):
        for j in range(0, m):
            if not (i == j or i + 1 == j) and abs(mat[i, j]) > 0.00001:
                return False
    return True


class TestDecompBidiag(test.TestCase):

    def test_decomp_bidiag(self):
        left, a, right = bd.decomp(A)
        self.assertTrue(is_bidiag(a))
        np.testing.assert_array_almost_equal(np.dot(np.dot(left, a), right), A)

    def test_is_bidiag(self):
        self.assertTrue(is_bidiag(B))
        self.assertFalse(is_bidiag(C))

if __name__ == '__main__':
    test.main()
