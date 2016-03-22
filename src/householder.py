import numpy as np


def householder(vec_x, vec_y):
    n = len(vec_x)

    if np.array_equal(vec_x, vec_y):
        return np.eye(n)

    mat_u = (vec_x - vec_y) / np.linalg.norm(vec_x - vec_y)

    mat_h = np.eye(n, n) - 2 * np.dot(np.array([mat_u]).transpose(), [mat_u])
    return mat_h


def householder_mul_vect(x, y, v):
    v = np.transpose([v])
    d = x - y
    d = [d]
    return v - 2 * np.dot((np.dot(np.transpose(d), d) / np.linalg.norm(d)**2), v)


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

'''
x = np.array([3, 4, 0])

y = np.array([0, 0, 5])

h = householder(x, y)

m = np.array([[2, 1, 0],
              [0, 2, 1],
              [1, 0, 0]])

v = np.array([1, 2, 0])

print np.dot(h, np.transpose([v]))
print householder_mul_vect(x, y, v)
print np.dot(h, m)
print householder_mul_mat_g(x, y, m)
'''