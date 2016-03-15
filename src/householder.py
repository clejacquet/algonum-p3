import numpy as np



'''
def transpose_colomn(c):
    n = len(c)
    res = []
    for i in range(n):
        res.append(c[i][0])
    return np.array(res)
print(transpose_colomn(Y))
'''



'''
print(np.dot(np.array(([1],
                      [2])),np.array([2,2])))
print(U,transpose_colomn(U),"le produit",np.dot(U,transpose_colomn(U)))
'''


def householder(vec_x, vec_y):
    n = len(vec_x)
    mat_u = (vec_x - vec_y) / np.linalg.norm(vec_x - vec_y)
    mat_h = np.eye(n, n) - 2 * np.dot(mat_u, mat_u.transpose())
    return mat_h
