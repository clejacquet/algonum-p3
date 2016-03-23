# coding=utf8

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import decomp_bidiag as db
import svd
import imgutils as img


def compression_couleur(a, k):
    """
    Prend une matrice a correspondant à une seule composante et un entier k et compresse cette couleur à l'ordre k
    :param a: matrice des valeurs d'une composante de couleur
    :param k: rang de la compression
    :return: la matrice a compressée
    """

    l, bd, r = db.decomp_opti(a)
    u, s, v = svd.SVD(bd)

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


def compression_couleur_opti(a, k):
    """
    Similaire à "compression_couleur", mais utilise les fonctions numpy, plus performantes
    :param a: matrice des valeurs d'une composante de couleur
    :param k: rang de la compression
    :return: la matrice a compressée
    """

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
            if res[i, j] < 0:
                res[i, j] = 0

    return res


# compression
def compression(a, k):
    """
    Effectue la compression au rang k de la matrice de triplets a
    :param a: matrice représentant une image à compresser
    :param k: rang de la compression
    :return: matrice de l'image compressée
    """

    r, g, b = img.decomposition_couleurs(a)
    r = compression_couleur_opti(r, k)
    g = compression_couleur_opti(g, k)
    b = compression_couleur_opti(b, k)
    return img.reconstruction_couleurs(r, g, b)


def compression_test(img_filename):
    """
    Affiche les différences de compression en changeant k sur une image
    :param img_filename: nom du fichier de l'image à évaluer
    """
    img_full = mpimg.imread(img_filename)
    img_comp5 = compression(img_full, 5)
    img_comp50 = compression(img_full, 50)
    img_comp100 = compression(img_full, 100)

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(img_full, interpolation='none')
    ax[0, 0].set_title('sans compression')
    ax[0, 1].imshow(img_comp5, interpolation='none')
    ax[0, 1].set_title('k = 5')

    ax[1, 0].imshow(img_comp50, interpolation='none')
    ax[1, 0].set_title('k = 50')
    ax[1, 1].imshow(img_comp100, interpolation='none')
    ax[1, 1].set_title('k = 100')

    plt.suptitle("Compression de \"" + img_filename + "\"")
    plt.show()


if __name__ == '__main__':
    compression_test("res/p3_takeoff_base.png")
    compression_test("res/p3_earth_base.png")

