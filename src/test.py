import numpy as np
import exo2

A = np.array([[1, 2, 3, 4],
              [7, 3, 9, 2],
              [3, 0, 4, 5]])


U, S, V = exo2.SVD(A)

print np.round(np.dot(np.dot(U, S), V))




