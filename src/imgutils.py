# coding=utf8

import numpy as np

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


# reconstruction_couleurs prend 3 matrices et construit une matrice de triplets (opération inverse de decomposition_couleurs)
def reconstruction_couleurs(r, g, b):
    n = np.shape(r)[0]
    m = np.shape(r)[1]
    a = np.zeros((n, m, 3))
    for i in range(n):
        for j in range(m):
            a[i][j][0] = r[i][j]
            a[i][j][1] = g[i][j]
            a[i][j][2] = b[i][j]
    return a