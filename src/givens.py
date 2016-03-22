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

def q(A):
    n = len(A)
    q = eye(n)
    for i in range (1, n):
        q = G(i, i-1, A) * q
        A = G(i, i-1, A) * A
    return q
'''
def test():
    A = np.array([[6, 0, 0, 0],
                  [18, 25, 0, 0],
                  [0, 14, 9, 0],
                  [0, 0, 7, 66]])
    Q = q(A)
    R = Q * A
    n = len(R)
    for i in range(0, n):
        for j in range(0, i):
            if R[i,j] < 10**(-5):
                R[i,j] = 0
    print("Q")
    print(Q)
    print("tQ*Q")
    print(np.round(np.transpose(Q)*Q,3))
    print("R")
    print(R)
    print("tQ*R")
    print(np.transpose(Q)*R)

test()
'''
