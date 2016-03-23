# coding=utf8

import numpy as np
import matplotlib.image as mpimg
import exo2
import time

'''
def comp_test():
    img_full = mpimg.imread("p3_takeoff_base.png")
    r, v, b = decomposition_couleurs(img_full)

    start = time.time()
    exo2.decomp_opti(r)
    end = time.time()
    print(end - start)


# decomposition_couleurs prend une matrice de triplets RGB et la décompose en 3 matrices avec les valeurs correspondant aux 3 couleurs
def decomposition_couleurs(a):
    n = np.shape(a)[0]
    m = np.shape(a)[1]
    r = np.eye(n, m)
    g = np.eye(n, m)
    b = np.eye(n, m)
    for i in range(n):
        for j in range(m):
            r[i][j] = a[i][j][0]
            g[i][j] = a[i][j][1]
            b[i][j] = a[i][j][2]
    return r, g, b

comp_test()'''

a = np.array([[1, 1, 1, 1],
              [1, 1, 1, 1],
              [1, 1, 1, 1],
              [1, 1, 1, 1]])

b = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 2, 0],
              [0, 0, 0, 2]])

print np.dot(a, b)

sub_a = a[:, 2:4]
sub_b = b[2:4]
print sub_a
print sub_b
ab = np.dot(sub_a, sub_b)

a[:, 2:4] = ab[:, 2:4]

print a