import numpy as np
import copy

import Householder as HH

NMax = 1024

def const_vect(sens, i, n, m, B_Diag):
    if (sens == 'v'):
        vect = B_Diag[i:n][i]
        vect_arriv = np.zeros(n-i+1)
        vect_arriv[0] = np.linalg.norm(vect)
    else:
        vect = B_Diag[i][(i+1):m]
        vect_arriv = np.zeros(m-i)
        vect_arriv[0] = np.linalg.norm(vect)
    return (vect, vect_arriv)

def Decomp_Bidiag(A):
    n = np.shape(A)[0]
    m = np.shape(A)[1]
    Qleft = np.eye(n)
    Qright = np.eye(m)
    B_Diag = copy.copy(A)

    for i in range(n):
        vect1, vect_arriv1 = const_vect('v',i,n,m,B_Diag)
        Q1 = HH.Householder(vect1, vect_arriv1)
        Qleft = np.dot(Qleft, Q1)
        B_Diag = np.dot(Q1, B_Diag)

        if (i != m-2):
            vect2, vect_arriv2 = const_vect('h',i,n,m,B_Diag)
            Q2 = HH.Householder(vect2,vect_arriv2)
            Qright = np.dot(Q2, Qright)
            B_Diag = np.dot(B_Diag, Q2)

        assert(np.dot(Qleft, np.dot(B_Diag, Qright)) == A)
        print(B_Diag)

    return (Qleft, B_Diag, Qright)

A = np.array([[1,2,3,4],[7,3,9,2],[3,0,4,5]])
print(A)
print("Decomp_Bidiag:")
print(Decomp_Bidiag(A)[1])

def SVD(A):
    n = np.shape(A)[0]
    U = np.eye(n)
    V = np.eye(n)
    S = Decomp_Bidiag(A)
    BD = Decomp_Bidiag(A)

    for i in range(0,NMax):
        (Q1, R1) = np.linalg.qr(np.transpose(S))
        (Q2, R2) = np.linalg.qr(np.transpose(R1))
        S = R2
        U = np.dot(U, Q2)
        V = np.dot(np.transpose(Q1), V)

        assert(np.dot(U, np.dot(S, V)) == BD)

    return (U, S, V)

def est_diag(A):
    n = np.shape(A)[0]
    m = np.shape(A)[1]
    for i in range(0,n):
        for j in range(0,m):
            if (i!=j and A[i][j]!=0):
                return False
    return True


