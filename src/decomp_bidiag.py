# coding=utf8

import numpy as np
import householder as HH


def extract_column(a, n, i):
    """
    Prend une matrice a et renvoie les éléments i à n de la colonne i
    :param a: matrice
    :param n: nombre de lignes de la matrice a
    :param i: numéro de la colonne à extraire
    :return: un vecteur vertical
    """

    return np.transpose([a[i:n, i]])


def extract_line(a, m, i):
    """
    Prend une matrice a et renvoie les éléments i+1 à m de la ligne i
    :param a: matrice
    :param m: nombre de colonnes de la matrice a
    :param i: numéro de la ligne à extraire
    :return: un vecteur vertical
    """

    return np.transpose([a[i, (i+1):m]])


def singlify_vector(v):
    """
    Prend un vecteur v et renvoie un vecteur de la même longueur contenant la norme de v comme premier élément, suivi de zéros
    :param v: vecteur
    :return: un vecteur
    """

    new_v = np.zeros(np.shape(v))
    new_v[0, 0] = np.linalg.norm(v)
    return new_v


def construct_householder(x):
    """
    Prend un vecteur x et renvoie la matrice de Householder qui envoie x sur le vecteur contenant la norme de x suivie de zéros
    :param x: vecteur
    :return: matrice carrée de la taille du vecteur x
    """

    y = singlify_vector(x)
    return HH.householder(x, y)


def resize_mat(mat, n):
    """
    Prend une matrice mat et un entier n et renvoie une matrice de taille n dont les éléments sont ceux de mat, complétée avec la matrice identité
    :param mat: matrice
    :param n: entier
    :return: matrice de taille n si n>taille(mat), mat sinon
    """
    n0 = np.shape(mat)[0]
    if n0 == n:
        return mat

    new_mat = np.eye(n)
    for i in range(0, n0):
        for j in range(0, n0):
            new_mat[(n - n0) + i, (n - n0) + j] = mat[i, j]
    return new_mat


def resize_vec(vec, n):
    """
    Prend un vecteur vec et un entier n et renvoie un vecteur de taille n dont les derniers éléments sont ceux de vec, et les autres sont des zéros
    :param vec: vecteur
    :param n: entier
    :return: vecteur de taille n si n>len(vec), vec sinon
    """
    n0 = np.shape(vec)[0]
    if n0 == n:
        return vec
    new_vec = np.zeros((n, 1))
    for i in range(0, n0):
        new_vec[(n - n0) + i, 0] = vec[i, 0]
    return new_vec


def MatrixMul( mtx_a, mtx_b):
    """
    Prend deux matrices mtx_a et mtx_b et renvoie le produit des deux matrices
    :param mtx_a: matrice
    :param mtx_b: matrice
    :return: matrice, erreur si les tailles des entrées ne correspondent pas
    """
    tpos_b = zip( *mtx_b)
    rtn = [[ sum( ea*eb for ea,eb in zip(a,b)) for b in tpos_b] for a in mtx_a]
    return rtn


def decomp_bad(a):
    """
    Prend une matrice a et renvoie un triplet de matrices dont le produit vaut a et dont la matrice centrale est bidiagonale
    :param a: matrice
    :return: une matrice carrée, une matrice bidiagonale et une matrice carrée
    """
    n, m = np.shape(a)
    left = np.eye(n)
    right = np.eye(m)

    bd = a

    for i in range(0, min(n, m)):
        q1 = resize_mat(construct_householder(extract_column(bd, n, i)), n)
        left = MatrixMul(left, q1)
        bd = MatrixMul(q1, bd)

        if i <= m - 2:
            q2 = resize_mat(construct_householder(extract_line(bd, m, i)), m)
            right = MatrixMul(q2, right)
            bd = MatrixMul(bd, q2)

        np.testing.assert_array_almost_equal(np.dot(np.dot(left, bd), right), a)

    return left, bd, right


def decomp_opti(a):
    """
    Même chose que decomp_bad mais avec des fonctions optimisées de multiplication de matrices
    :param a: matrice
    :return: une matrice carrée, une matrice bidiagonale et une matrice carrée
    """
    n, m = np.shape(a)
    left = np.eye(n)
    right = np.eye(m)

    bd = a

    for i in range(0, min(n, m)):
        q1_x = extract_column(bd, n, i)
        q1_y = singlify_vector(q1_x)
        q1_x = resize_vec(q1_x, n)
        q1_y = resize_vec(q1_y, n)

        left = HH.householder_mul_mat_g(q1_x, q1_y, left)
        bd = HH.householder_mul_mat_d(q1_x, q1_y, bd)

        if i <= m - 2:
            q2_x = extract_line(bd, m, i)
            q2_y = singlify_vector(q2_x)
            q2_x = resize_vec(q2_x, m)
            q2_y = resize_vec(q2_y, m)

            right = HH.householder_mul_mat_d(q2_x, q2_y, right)
            bd = HH.householder_mul_mat_g(q2_x, q2_y, bd)

        np.testing.assert_array_almost_equal(np.dot(np.dot(left, bd), right), a)

    return left, bd, right


if __name__ == '__main__':
    A = np.array([[1,2,3,4],
                  [7,3,9,2],
                  [3,0,4,5]])
    print(A)
    print("Decomp_Bidiag:")
    print(np.round(decomp_opti(A)[1]), 3)
    print("\n")
