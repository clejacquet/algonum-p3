# coding=utf8

import numpy as np
import householder as HH


def extract_column(a, n, i):
    """

    :param a:
    :param n:
    :param i:
    :return:
    """

    return np.transpose([a[i:n, i]])


def extract_line(a, m, i):
    """

    :param a:
    :param m:
    :param i:
    :return:
    """

    return np.transpose([a[i, (i+1):m]])


def singlify_vector(v):
    """

    :param v:
    :return:
    """

    new_v = np.zeros(np.shape(v))
    new_v[0, 0] = np.linalg.norm(v)
    return new_v


def construct_householder(x):
    """

    :param x:
    :return:
    """

    y = singlify_vector(x)
    return HH.householder(x, y)


def resize_mat(mat, n):
    """

    :param mat:
    :param n:
    :return:
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

    :param vec:
    :param n:
    :return:
    """
    n0 = np.shape(vec)[0]
    if n0 == n:
        return vec
    new_vec = np.zeros((n, 1))
    for i in range(0, n0):
        new_vec[(n - n0) + i, 0] = vec[i, 0]
    return new_vec


def decomp_bad(a):
    """

    :param a:
    :return:
    """
    n, m = np.shape(a)
    left = np.eye(n)
    right = np.eye(m)

    bd = a

    for i in range(0, min(n, m)):
        q1 = resize_mat(construct_householder(extract_column(bd, n, i)), n)
        left = np.dot(left, q1)
        bd = np.dot(q1, bd)

        if i <= m - 2:
            q2 = resize_mat(construct_householder(extract_line(bd, m, i)), m)
            right = np.dot(q2, right)
            bd = np.dot(bd, q2)

        np.testing.assert_array_almost_equal(np.dot(np.dot(left, bd), right), a)

    return left, bd, right


def decomp_opti(a):
    """

    :param a:
    :return:
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
