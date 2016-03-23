# coding=utf8

import numpy as np


def decomposition_couleurs(a):
    """
    Prend une matrice de triplets RGB et la décompose en 3 matrices avec les valeurs correspondant aux 3 couleurs
    :param a: matrice de triplets
    :return: trois matrices contenant les éléments des triplets de a
    """
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


# reconstruction_couleurs
def reconstruction_couleurs(r, g, b):
    """
    Prend 3 matrices et construit une matrice de triplets (opération inverse de decomposition_couleurs)
    :param r: matrice
    :param g: matrice
    :param b: matrice
    :return: matrice de triplets dont les éléments sont les éléments des matrices en entrée
    """

    n = np.shape(r)[0]
    m = np.shape(r)[1]
    a = np.zeros((n, m, 3))
    for i in range(n):
        for j in range(m):
            a[i][j][0] = r[i][j]
            a[i][j][1] = g[i][j]
            a[i][j][2] = b[i][j]
    return a
