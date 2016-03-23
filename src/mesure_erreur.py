import numpy as np
from math import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import compression as cp

valeur_max = 255
rayon_ssim = 8

def PSNR_couleur(A_orig, A_compres, couleur):
    """
    Prend deux matrices A_orig et A_compres et un entier couleur et renvoie le PSNR correspondant à la couleur précisée (0:rouge, 1:vert, 2:bleu)
    :param A_orig: matrice de triplets (image originale)
    :param A_compres: matrice de triplets (image compressée)
    :param couleur: entier valant 0, 1 ou 2
    :return: réel
    """

    e = 0
    n, m = np.shape(A_orig)
    for i in range(n):
        for j in range(m):
            e += (A_orig[i,j,couleur] - A_compres[i,j,couleur])**2
    e = e/(n*m)
    return 10*log(((valeur_max)**e)/e)


def PSNR(A_orig, A_compres):
    """
    Prend deux matrices A_orig et A_compres et un entier couleur et renvoie la somme des PSNR de chaque couleur
    :param A_orig: matrice de triplets (image originale)
    :param A_compres: matrice de triplets (image compressée)
    :return: réel
    """

    return PSNR_couleur(A_orig, A_compres, 0) + PSNR_couleur(A_orig, A_compres, 1) + PSNR_couleur(A_orig, A_compres, 2)


def moy_ssim(A, i, j, couleur):
    """
    Prend une matrice A, deux entiers i et j et un entier couleur et renvoie la moyenne des valeurs de A autour de l'élément A[i,j] pour la couleur précisée (0:rouge, 1:vert, 2:bleu)
    :param A: matrice de 
    :param i: entier
    :param j: entier
    :param couleur: entier valant 0, 1 ou 2
    :return: réel
    """

    moy = 0
    n, m, p = np.shape(A)
    compt = 0
    for k in range(i-rayon_ssim, i+rayon_ssim):
        for l in range(j-rayon_ssim, j+rayon_ssim):
            if 0<=k<n and 0<=l<m:
                moy += A[k,l,couleur]
                compt += 1
    return moy/compt
    

def sigma_ssim(A, i, j, couleur):
    """
    Prend une matrice A, deux entiers i et j et un entier couleur et renvoie la variance des valeurs de A autour de l'élément A[i,j] pour la couleur précisée (0:rouge, 1:vert, 2:bleu)
    :param A: matrice de triplets
    :param i: entier
    :param j: entier
    :param couleur: entier valant 0, 1 ou 2
    :return: réel
    """

    sigma = 0
    n, m, p = np.shape(A)
    compt = 0
    for k in range(i-rayon_ssim, i+rayon_ssim):
        for l in range(j-rayon_ssim, j+rayon_ssim):
            if 0<=k<n and 0<=l<m:
                sigma += (A[k, l, couleur] - moy_ssim(A, i, j, couleur))**2
                compt += 1
    return sigma/compt


def covar_ssim(A, B, i, j, couleur):
    """
    Prend deux matrices A et B, deux entiers i et j et un entier couleur et renvoie la covariance des valeurs de A et de B autour du point (i,j) pour la couleur précisée (0:rouge, 1:vert, 2:bleu)
    :param A: matrice de triplets
    :param B: matrice de triplets
    :param i: entier
    :param j: entier
    :param couleur: entier valant 0, 1 ou 2
    :return:
    """

    covar = 0
    n, m, p = np.shape(A)
    compt = 0
    for k in range(i-rayon_ssim, i+rayon_ssim):
        for l in range(j-rayon_ssim, j+rayon_ssim):
            if 0<=k<n and 0<=l<m:
                covar += (A[k, l, couleur] - moy_ssim(A, i, j, couleur))*(B[k, l, couleur] - moy_ssim(B, i, j, couleur))
                compt += 1
    return covar/compt


def SSIM_couleur(A_orig, A_compres, couleur):
    """
    Prend deux matrices A_orig et A_compres et un entier couleur et renvoie une matrice correspondant à la carte des différences entre les deux matrices pour la couleur précisée (0:rouge, 1:vert, 2:bleu)
    :param A_orig: matrice de triplets (image originale)
    :param A_compres: matrice de triplets (image compressée)
    :param couleur: entier valant 0, 1 ou 2
    :return: matrice
    """

    n, m, p = np.shape(A_compres)
    carte_ssim = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            moy_orig = moy_ssim(A_orig, i, j, couleur)
            moy_compres = moy_ssim(A_compres, i, j, couleur)
            sigma_orig = sigma_ssim(A_orig, i, j, couleur)
            sigma_compres = sigma_ssim(A_compres, i, j, couleur)
            covar = covar_ssim(A_orig, A_compres, i, j, couleur)
            carte_ssim[i, j] = ((2*moy_orig*moy_compres + 6.5025)*(2*covar + 58.5225))/(((moy_orig)**2 + (moy_compres)**2 + 6.5025)*((sigma_orig)**2 + (sigma_compres)**2 + 58.5225))
    return carte_ssim


def SSIM(A_orig, A_compres):
    """
    Prend deux matrices A_orig et A_compres et renvoie une matrice correspondant à la carte des différences entre les deux matrices
    :param A_orig: matrice de triplets (image originale)
    :param A_compres: matrice de triplets (image compressée)
    :return: matrice
    """

    carte0 = SSIM_couleur(A_orig, A_compres, 0)
    carte1 = SSIM_couleur(A_orig, A_compres, 1)
    carte2 = SSIM_couleur(A_orig, A_compres, 2)
    return carte0 + carte1 + carte2


if __name__ == '__main__':

    img_full = mpimg.imread("p3_takeoff_base.png")
    img_compres = cp.compression(img_full, 50)
    ssim = SSIM(img_full, img_compres)
    plt.imshow(ssim, interpolation = 'none')
    plt.show()

