import numpy as np
X = np.array(([3],
             [4],
             [0]))
Y = np.array(([0],
             [0],
             [5]))
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
H = np.array([[0.64,-0.48,0.6]
            ,[-0.48,0.36,0.8]
            ,[0.6,0.8,0]])
'''

'''
print(np.dot(np.array(([1],
                      [2])),np.array([2,2])))
print(U,transpose_colomn(U),"le produit",np.dot(U,transpose_colomn(U)))
'''

def Householder(X,Y):
    n=len(X)
    U = (X-Y)/np.linalg.norm(X-Y)
    H = np.eye(n,n) -2*np.dot(U,U.transpose())
    print(H)


Householder(X,Y)
