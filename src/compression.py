# coding=utf8

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import exo2 as ex2


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


# compression_couleur prend une matrice a correspondant à une seule couleur et un entier k et compresse cette couleur à l'ordre k
def compression_couleur2(a, k):
    l, bd, r = ex2.decomp_opti(a)
    u, s, v = ex2.SVD(bd)

    n, m = np.shape(a)
    for i in range(k + 1, min(n, m)):
        s[i, i] = 0

    res_interm = np.dot(np.dot(u, s), v)
    res = np.dot(np.dot(l, res_interm), r)

    for i in range(0, n):
        for j in range(0, m):
            if res[i, j] > 1:
                res[i, j] = 1

    return res


def compression_couleur(a, k):
    u, s, v = np.linalg.svd(a)
    n, m = np.shape(a)
    for i in range(k + 1, min(n, m)):
        s[i] = 0

    s = np.dot(np.diag(s), np.eye(np.shape(a)[0], np.shape(a)[1]))

    res = np.dot(np.dot(u, s), v)

    for i in range(0, n):
        for j in range(0, m):
            if res[i, j] > 1:
                res[i, j] = 1

    return res


# compression effectue la compression à l'orde k de la matrice de triplets a
def compression(a, k):
    r, g, b = decomposition_couleurs(a)
    r = compression_couleur(r, k)
    print("r:")
    print(r)
    g = compression_couleur(g, k)
    print("g:")
    print(g)
    b = compression_couleur(b, k)
    print("b:")
    print(b)
    return reconstruction_couleurs(r, g, b)


def comp_test(k):
    img_full = mpimg.imread("p3_takeoff_base.png")
    img_comp = compression(img_full, k)
    plt.subplot(1, 2, 1)
    plt.imshow(img_full, interpolation='none')
    plt.title("Image originale")

    plt.subplot(1, 2, 2)
    plt.imshow(img_comp, interpolation='none')
    plt.title("Compression rang "+str(k))

    plt.show()
    return True

comp_test(5)

# test = np.array([[[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]],
#                  [[0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2]],
#                  [[0.3, 0.3, 0.3], [0.3, 0.3, 0.3], [0.3, 0.3, 0.3]],
#                  [[0.4, 0.4, 0.4], [0.4, 0.4, 0.4], [0.4, 0.4, 0.4]]])

# print("compression:")
# print(compression(test, 2))
# plt.subplot(1, 2, 1)
# plt.imshow(test)
# plt.subplot(1, 2, 2)
# plt.imshow(compression(test, 2))
# plt.show()
