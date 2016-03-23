from scipy import *
import numpy as np

def c(p,q,A): 
    """cosinus theta""" 
    if A[p,p]==0 and A[q,p]==0:
        return 1
    else: 
        return A[p,p]/sqrt(A[p,p]*A[p,p]+A[q,p]*A[q,p]) 
 
def s(p,q,A): 
    """sinus theta""" 
    if A[p,p]==0 and A[q,p]==0:
        return 0
    else: 
        return A[q,p]/sqrt(A[p,p]*A[p,p]+A[q,p]*A[q,p]) 
 
def G(q,p,A): 
    """Selon l'algo""" 
    G=eye(len(A)) 
    G[p,p]=c(p,q,A) 
    G[p,q]=s(p,q,A) 
    G[q,p]= -s(p,q,A) 
    G[q,q]= c(p,q,A) 
    return matrix(G) 

def Q(A):
    """

    :param A:
    :return:
    """

    n, m = shape(A)
    q = eye(n)
    for i in range(1, min(n, m + 1)):
        q = G(i, i-1, A) * q
        A = G(i, i-1, A) * A
    return q


def qr(A):
    """

    :param A:
    :return:
    """

    q = Q(A)
    r = np.dot(q, A)
    return q, r


if __name__ == '__main__':
    A = np.array([[6, 0, 0, 0, 0, 0],
                  [18, 25, 0, 0, 0, 0],
                  [0, 14, 9, 0, 0, 0],
                  [0, 0, 7, 66, 0, 0]])

    B = np.array([[6, 0, 0, 0],
                  [18, 25, 0, 0],
                  [0, 14, 9, 0],
                  [0, 0, 7, 66],
                  [0, 0, 0, 12],
                  [0, 0, 0, 0]])
    Q, R = qr(A)

    print("Q")
    print(Q)
    print("tQ*Q")
    print(np.round(np.transpose(Q)*Q,3))
    print("R")
    print(np.round(R))
    print("tQ*R")
    print(np.round(np.transpose(Q)*R))


