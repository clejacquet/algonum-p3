import numpy as np


def householder(vec_x, vec_y):
    vec_x = np.transpose([vec_x])
    vec_y = np.transpose([vec_y])

    print vec_x
    print vec_y

    n = len(vec_x)

    if np.array_equal(vec_x, vec_y):
        return np.eye(n)

    mat_u = (vec_x - vec_y) / np.linalg.norm(vec_x - vec_y)
    mat_h = np.eye(n, n) - 2 * np.dot(mat_u, mat_u.transpose())
    return mat_h


def householder2(vec_x, vec_y):
    n = len(vec_x)

    if np.array_equal(vec_x, vec_y):
        return np.eye(n)

    mat_u = (vec_x - vec_y) / np.linalg.norm(vec_x - vec_y)
    mat_h = np.eye(n, n) - 2 * np.dot(np.transpose(mat_u), mat_u)
    return mat_h


def householder_mul_vect(x, y, v):
    d = x - y
    v_t = np.transpose([v])
    return (v_t - 2 * np.dot((np.dot(d, np.transpose(d)) / np.linalg.norm(d)**2), v_t))[:, 0]


def householder_mul_mat_d(x, y, mat):
    n, m = np.shape(mat)
    mat_t = np.transpose(mat)
    res = np.empty((m, n))

    for i in range(0, m):
        res[i] = householder_mul_vect(x, y, mat_t[i])

    np.transpose(res)
    return res


def householder_mul_mat_g(x, y, mat):
    return np.transpose(householder_mul_mat_d(x, y, np.transpose(mat)))


x = np.array([3, 4, 0])
y = np.array([0, 0, 5])

h = householder(x, y)

v = np.array([[2, 1, 0],
              [0, 2, 1],
              [1, 0, 0]])

print np.dot(v, h)
print householder_mul_mat_g(x, y, v)
