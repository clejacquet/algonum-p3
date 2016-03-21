import numpy as np
import householder as HH

NMax = 1024


def extract_column(a, n, i):
    return np.transpose([a[i:n, i]])


def extract_line(a, m, i):
    return np.transpose([a[i, (i+1):m]])


def singlify_vector(v):
    new_v = np.zeros(np.shape(v))
    new_v[0, 0] = np.linalg.norm(v)
    return new_v


def construct_householder(x):
    y = singlify_vector(x)
    return HH.householder(x, y)


def resize(mat, n):
    n0 = np.shape(mat)[0]
    if n0 == n:
        return mat

    new_mat = np.eye(n)
    for i in range(0, n0):
        for j in range(0, n0):
            new_mat[(n - n0) + i, (n - n0) + j] = mat[i, j]
    return new_mat


def decomp(a):
    n, m = np.shape(a)
    left = np.eye(n)
    right = np.eye(m)

    bd = a

    for i in range(0, min(n, m)):
        q1 = resize(construct_householder(extract_column(bd, n, i)), n)
        left = np.dot(left, q1)
        bd = np.dot(q1, bd)

        if i <= m - 2:
            q2 = resize(construct_householder(extract_line(bd, m, i)), m)
            right = np.dot(q2, right)
            bd = np.dot(bd, q2)

        np.testing.assert_array_almost_equal(np.dot(np.dot(left, bd), right), a)

    return left, bd, right


# def decomp2(a):
#     n, m = np.shape(a)
#     left = np.eye(n)
#     right = np.eye(m)
#
#     bd = a
#
#     for i in range(0, min(n, m)):
#         q1 = resize(construct_householder(extract_column(bd, n, i)), n)
#         left = np.dot(left, q1)
#         bd = np.dot(q1, bd)
#
#         if i <= m - 2:
#             q2 = resize(construct_householder(extract_line(bd, m, i)), m)
#             right = np.dot(q2, right)
#             bd = np.dot(bd, q2)
#
#         # np.testing.assert_array_almost_equal(np.dot(np.dot(left, bd), right), a)
#
#     return left, bd, right

# A = np.array([[1,2,3,4],
#               [7,3,9,2],
#               [3,0,4,5]])
# print(A)
# print("Decomp_Bidiag:")
# print(np.round(decomp(A)[1]), 3)
# print("\n")


def SVD(A):
    n = np.shape(A)[0]
    m = np.shape(A)[1]
    U = np.eye(n)
    V = np.eye(m)
    S = decomp(A)[1]
    BD = decomp(A)[1]

    for i in range(0,NMax):
        Q1, R1 = np.linalg.qr(np.transpose(S))
        Q2, R2 = np.linalg.qr(np.transpose(R1))
        S = R2
        U = np.dot(U, Q2)
        V = np.dot(np.transpose(Q1), V)

        np.testing.assert_array_almost_equal(np.dot(U, np.dot(S, V)), BD)

    return U, S, V


def est_diag(A):
    n = np.shape(A)[0]
    m = np.shape(A)[1]
    for i in range(0,n):
        for j in range(0,m):
            if (i!=j and A[i][j]!=0):
                return False
    return True

# print(SVD(A)[1])

