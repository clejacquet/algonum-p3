# -*- coding: utf-8 -*-
import numpy as np
import random
import time


def householder(vec_x, vec_y):
    """
    Prend vec_x et vec_y de même taille et renvoie la matrice de Householder mat_h telle que mat_h*vec_x = vec_y
    :param vec_x: vecteur de taille n
    :param vec_y: vecteur de taille n
    :return: matrice de Householder de taille n*n nécesseaire à la transformation de vec_x en vec_y
    """

    n = len(vec_x)

    if np.array_equal(vec_x, vec_y):
        return np.eye(n)

    mat_u = (vec_x - vec_y) / np.linalg.norm(vec_x - vec_y)
    mat_h = np.eye(n, n) - 2 * np.dot(mat_u, mat_u.transpose())
    return mat_h


def householder_mul_vect_d(x, y, v):
    """
    Prend x, y et v de même taille et renvoie la transformation de Householder de v par la matrice de Householder transformant x en y
    :param x: vecteur
    :param y: vecteur
    :param v: vecteur
    :return: transformée de Householder optimisée de v par la matrice de Householder transformant x en y
    """

    d = x - y
    return v - 2 * np.dot((np.dot(d, np.transpose(d)) / np.linalg.norm(d)**2), v)


def householder_mul_mat_d(x, y, mat):
    """
    Prend x et y vecteurs et mat une matrice en paramètres et renvoie la transformée de Householder à droite 
    de mat par la matrice de Householder transformant x en y
    :param x: vecteur
    :param y: vecteur
    :param mat: matrice
    :return: transformée de Householder optimisée de m par la matrice de Householder transformant x en y
    """

    n, m = np.shape(mat)
    res = np.empty((n, m))

    for i in range(0, m):
        res[:, i] = householder_mul_vect_d(x, y, mat[:, i])
    return res


def householder_mul_mat_g(x, y, mat):
    """
    Prend x et y vecteurs et mat une matrice en paramètres et renvoie la transformée de Householder à gauche 
    de mat par la matrice de Householder transformant x en y
    :param x: vecteur
    :param y: vecteur
    :param mat: matrice
    :return: produit optimisé de la matice de Householder et mat
    """

    return np.transpose(householder_mul_mat_d(x, y, np.transpose(mat)))


def matrix_gen(n):
    """
    Génère une matrice m de taille n*n aléatoirement
    :param n: entier
    :return: matrice aléatoire de taille n
    """

    m = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            m[i][j] = random.random()
    return m


def vec_gen(n):
    """
    Génère un vecteur v de taille n aléatoirement
    :param n: entier
    :return: vecteur aléatoire de taille n
    """

    v = np.zeros(n)
    for i in range(n):
        v[i] = random.random()
    return v


def prod_mat(A, B):
    """
    Réalise le produit d'une matrice A par une matrice B (np.dot étant trop puissant, permet de comparer le produit matriciel au produit optimisé.
    :param A: matrice
    :param B: matrice
    :return: A*B
    """

    n = len(A)
    C = np.zeros((n,n))
    if n == len(B):
        for i in range(n):
            for j in range(n):
                s=0
                for k in range(n):
                    s+= A[i][k]*B[k][j]
                C[i][j]=s
    return C


def complexity_graph():
    """
    Dresse les graphes comparatifs des complexités des produits optimisés et non-optimisés pour des vecteurs et matrices
    :return: graphes des complexités
    """

    n = 50
    tab_x=range(1,n)
    tab_y_vec_opti=[]
    tab_y_vec= []
    tab_y_mat_opti=[]
    tab_y_mat=[]
    for j in tab_x:
        X = vec_gen(j)
        Y = vec_gen(j)
        V = vec_gen(j)
        M = matrix_gen(j)
        t1 = time.time()
        H = householder(X,Y)
        np.dot(H,V)
        t2 = time.time()
        householder_mul_vect_d(X,Y,V)
        t3 = time.time()
        H = householder(X,Y)
        prod_mat(H,M)
        t4 = time.time()
        householder_mul_mat_d(X,Y,M)
        t5 = time.time()
        tab_y_vec.append(t2-t1)
        tab_y_vec_opti.append(t3-t2)
        tab_y_mat.append(t4-t3)
        tab_y_mat_opti.append(t5-t4)
    return tab_x,tab_y_vec_opti,tab_y_vec,tab_y_mat_opti,tab_y_mat
