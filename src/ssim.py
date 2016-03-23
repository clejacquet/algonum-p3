# coding=utf-8
import numpy as np
from scipy import signal
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import compression as cp
from src.imgutils import decomposition_couleurs


def fspecial_gauss(size, sigma):
    """
    FONCTION PRISE EN LIGNE POUR UN CALCUL OPTIMISÉ DU SSIM
    Function to mimic the 'fspecial' gaussian MATLAB function

    Crédit: https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()


def ssim_map(img1, img2, cs_map=False):
    """
    FONCTION PRISE EN LIGNE POUR UN CALCUL OPTIMISÉ DU SSIM

    Return the Structural Similarity Map corresponding to input images img1
    and img2 (images are assumed to be uint8)

    This function attempts to mimic precisely the functionality of ssim.m a
    MATLAB provided by the author's of SSIM
    Crédit: https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255 #bitdepth of image
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = signal.fftconvolve(window, img1)
    mu2 = signal.fftconvolve(window, img2)
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(window, img1*img1) - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2*img2) - mu2_sq
    sigma12 = signal.fftconvolve(window, img1*img2) - mu1_mu2
    if cs_map:
        return (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        return ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))


def ssim(img1, img2):
    """

    :param map:
    :return:
    """

    s_map = SSIM_color(img1, img2)
    n, m = np.shape(s_map)
    sum = 0
    for i in range(0, n):
        for j in range(0, m):
            sum += s_map[i, j]
    return sum / (n * m)


def SSIM_color(img1, img2):
    """
    Réalise la carte SSIM des correspondances entre les images couleurs img1 et img2
    :param img1: première image sujette à la comparaison
    :param img2: seconde image sujette à la comparaison
    :return: la carte SSIM des correspondances
    """

    img1_r, img1_g, img1_b = decomposition_couleurs(img1)
    img2_r, img2_g, img2_b = decomposition_couleurs(img2)
    mapR = ssim_map(img1_r, img2_r)
    mapG = ssim_map(img1_g, img2_g)
    mapB = ssim_map(img1_b, img2_b)

    n, m = np.shape(mapR)
    full_carte = np.empty((n, m))

    for i in range(0, n):
        for j in range(0, m):
            full_carte[i, j] = (mapR[i, j] + mapG[i, j] + mapB[i, j]) / 3

    return full_carte


def SSIM_test(img_filename):
    """
    Réalise la comparaison des cartes de SSIM de la compression d'images à différents rang
    :param img_filename: nom de l'image de test
    """

    img_full = mpimg.imread(img_filename)
    img_comp5 = cp.compression(img_full, 5)
    img_comp50 = cp.compression(img_full, 50)
    img_comp100 = cp.compression(img_full, 100)

    ssim5 = SSIM_color(img_full, img_comp5)
    ssim50 = SSIM_color(img_full, img_comp50)
    ssim100 = SSIM_color(img_full, img_comp100)

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(ssim5, interpolation='none', cmap='gray')
    ax[0].set_title('k = 5')
    ax[1].imshow(ssim50, interpolation='none', cmap='gray')
    ax[1].set_title('k = 50')
    ax[2].imshow(ssim100, interpolation='none', cmap='gray')
    ax[2].set_title('k = 100')

    plt.suptitle("Cartes SSIM de \"" + img_filename + "\"")
    plt.show()

if __name__ == '__main__':
    #SSIM_test("res/p3_takeoff_base.png")
    #SSIM_test("res/p3_earth_base.png")
    img_full = mpimg.imread("res/p3_takeoff_base.png")
    img_full = img_full.astype(np.float64)
    tab_x = [10*i for i in range(2, 25)]
    tab_y = [ssim(img_full, cp.compression(img_full, k)) for k in tab_x]
    plt.plot(tab_x,tab_y,'-b')
    plt.suptitle("SSIM total, en fonction de k")
    plt.show()
    print ssim(img_full,cp.compression(img_full,10))
