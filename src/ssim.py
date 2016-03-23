import numpy as np
from scipy import signal
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import compression as cp
import skimage
import mesure_erreur as me

def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()


def ssim(img1, img2, cs_map=False):
    """Return the Structural Similarity Map corresponding to input images img1
    and img2 (images are assumed to be uint8)

    This function attempts to mimic precisely the functionality of ssim.m a
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
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


def decomposition_couleurs(a):
    """

    :param a:
    :return:
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


def SSIM_color(img1, img2):
    """

    :param img1:
    :param img2:
    :return:
    """

    img1_r, img1_g, img1_b = decomposition_couleurs(img1)
    img2_r, img2_g, img2_b = decomposition_couleurs(img2)
    carte0 = ssim(img1_r, img2_r)
    carte1 = ssim(img1_g, img2_g)
    carte2 = ssim(img1_b, img2_b)
    return carte0 + carte1 + carte2


def SSIM_test(img_filename):
    """

    :param img_filename:
    :return:
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
    img_full = mpimg.imread("res/belle-de-nuit-bicolore.jpg")
    tab_x = [10*i for i in range(15)]
    tab_y = [skimage.measure.ssim(img_full,cp.compression(img_full,k)) for k in tab_x]
    plt.plot(tab_x,tab_y,'-b')
    plt.show()
