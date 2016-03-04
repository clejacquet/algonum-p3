import numpy as np
import copy

def QR_facto(A):
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
