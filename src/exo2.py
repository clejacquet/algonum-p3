import numpy as np
import copy

import householder as HH

NMax = 1024


def const_vect(sens, i, n, m, B_Diag):
    if (sens == 'v'):
        vect = np.zeros((n-i,1))
        for j in range(i,n):
            vect[j-i][0] = B_Diag[j][i]
        vect_arriv = np.zeros((n-i,1))
        vect_arriv[0][0] = np.linalg.norm(vect)
    else:
        vect = np.zeros((m-i-1,1))
        for k in range(i+1,m):
            vect[k-i-1][0] = B_Diag[i+1][k]
        vect_arriv = np.zeros((m-i-1,1))
        vect_arriv[0][0] = np.linalg.norm(vect)
    return (vect, vect_arriv)


def resize(A,n):
    t = np.shape(A)[0]
    if (t >= n):
        return A
    else:
        ret = np.eye(n)
        for i in range(t):
            for j in range(t):
                ret[n-t+i][n-t+j] = A[i][j]
    return ret


def Decomp_Bidiag(A):
    n = np.shape(A)[0]
    m = np.shape(A)[1]

    Qleft = np.eye(n)
    Qright = np.eye(m)
    B_Diag = copy.copy(A)

    for i in range(0, n-1):
        vect1 = const_vect('v',i,n,m,B_Diag)[0]
        vect_arriv1 = const_vect('v',i,n,m,B_Diag)[1]
        Q1 = HH.householder(vect1, vect_arriv1)
        Q1 = resize(Q1,n)
        Qleft = np.dot(Qleft,Q1)
        B_Diag = np.dot(Q1, B_Diag)

        if (i < m-2):
            vect2 = const_vect('h',i,n,m,B_Diag)[0]
            vect_arriv2 = const_vect('h',i,n,m,B_Diag)[1]
            Q2 = HH.householder(vect2,vect_arriv2)
            Q2 = resize(Q2, m)
            Qright = np.dot(Q2, Qright)
            B_Diag = np.dot(B_Diag, Q2)

        np.testing.assert_array_almost_equal(np.dot(Qleft, np.dot(B_Diag, Qright)), A)
        print np.dot(Qleft, np.dot(B_Diag, Qright))
        print A

    return Qleft, B_Diag, Qright


A = np.array([[1,2,3,4],
              [7,3,9,2],
              [3,0,4,5]])
print(A)
print("Decomp_Bidiag:")
print(Decomp_Bidiag(A)[1])
print("\n")


def SVD(A):
    n = np.shape(A)[0]
    m = np.shape(A)[1]
    U = np.eye(n)
    V = np.eye(m)
    S = Decomp_Bidiag(A)[1]
    BD = Decomp_Bidiag(A)[1]

    for i in range(0,NMax):
        Q1, R1 = np.linalg.qr(np.transpose(S))
        Q2, R2 = np.linalg.qr(np.transpose(R1))
        S = R2
        U = np.dot(U, Q2)
        V = np.dot(np.transpose(Q1), V)

        #assert(np.array_equal(np.dot(U, np.dot(S, V)),BD))

    return U, S, V


def est_diag(A):
    n = np.shape(A)[0]
    m = np.shape(A)[1]
    for i in range(0,n):
        for j in range(0,m):
            if (i!=j and A[i][j]!=0):
                return False
    return True

print(SVD(A)[1])

