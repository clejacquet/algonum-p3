import numpy as np
import copy

NMax = 1024

def Decomp_Bidiag(A):
    n = np.shape(A)[0]
    m = np.shape(A)[1]
    Qleft = np.eye(n)
    Qright = np.eye(n)
    B_Diag = copy.copy(A)

    for i in range(n):
        #initialisation de Q1->Householder(V=B_Diag[i:n,i], [norm(V),0,...,0])
        Qleft = np.dot(Qleft, Q1)
        B_Diag = np.dot(Q1, B_Diag)

        if !(i == m-2):
            #initialisation de Q2->Householder(V=B_Diag[i,(i+1):m], [norm(V),0,...,0])
            Qright = np.dot(Q2, Qright)
            B_Diag = np.dot(B_Diag, Q2)

        assert(np.dot(Qleft, np.dot(B_Diag, Qright)) == A)
        print(B_Diag)

    return (Qleft, B_Diag, Qright)


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
            if (i!=j && A[i][j]!=0):
                return False
    return True


