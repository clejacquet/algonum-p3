# coding=utf8

import numpy as np

NMax = 1024


def modifSU(U,S):
    n = S.shape[0]
    diagS = []
    triS = []

    for i in range(0, n):
        diagS.append(S[i, i])
        triS.append(abs(S[i, i]))

    np.sort(triS)

    for i in range(0, n):
        diagS[i] /= triS[i]

    for i in range(0, n):
        S[i, i] = triS[i]
        U[:, i] *= diagS[i]

    return U, S

    '''
    diagS = np.empty([min(l,c)])
    for i in range(0,min(l,c)):
        diagS[i] = S[i,i]
    triS = np.empty([min(l,c)])
    triS = sorted(diagS)
    for i in range(0,min(l,c)):
        diagS[i] = float(diagS[i])/float(triS[i])
    for i in range(0,min(l,c)):
        S[i,i] = triS[i]
        U[:,i] = U[:,i]*float(diagS[i])

    return(U,S)
    '''

def resize_diag(a, n, m):
    mat = np.empty((n, m))
    for i in range(0, min(n, m)):
        mat[i, i] = a[i]
    return mat


def SVD(BD):
    n = np.shape(BD)[0]
    m = np.shape(BD)[1]
    U = np.eye(n)
    V = np.eye(m)

    S = BD

    for i in range(0, NMax):
        Q1, R1 = np.linalg.qr(np.transpose(S))
        Q2, R2 = np.linalg.qr(np.transpose(R1))
        S = R2
        U = np.dot(U, Q2)
        V = np.dot(np.transpose(Q1), V)

        np.testing.assert_array_almost_equal(np.dot(np.dot(U, S), V), BD)

    U, S = modifSU(U, S)

    return U, S, V

'''
A = np.array([[1,2,0,0],
              [0,3,9,0],
              [0,0,1,5]])


print(SVD(A)[0])
print("\n")
print(SVD(A)[1])
print("\n")
print(SVD(A)[2])
print("\n")
print(np.dot(np.dot(SVD(A)[0],SVD(A)[1]),SVD(A)[2]))
print("\n")
print(decomp_opti(A)[1])
'''
