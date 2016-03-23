# -*- coding: utf-8 -*-
import numpy as np
import random
import time
import matplotlib.pyplot as plt

def householder(vec_x, vec_y):
    n = len(vec_x)

    if np.array_equal(vec_x, vec_y):
        return np.eye(n)

    mat_u = (vec_x - vec_y) / np.linalg.norm(vec_x - vec_y)
    mat_h = np.eye(n, n) - 2 * np.dot(mat_u, mat_u.transpose())
    return mat_h


def householder_mul_vect_d(x, y, v):
    d = x - y
    return v - 2 * np.dot((np.dot(d, np.transpose(d)) / np.linalg.norm(d)**2), v)


def householder_mul_mat_d(x, y, mat):
    n, m = np.shape(mat)
    res = np.empty((n, m))

    for i in range(0, m):
        res[:, i] = householder_mul_vect_d(x, y, mat[:, i])
    return res


def householder_mul_mat_g(x, y, mat):
    return np.transpose(householder_mul_mat_d(x, y, np.transpose(mat)))

def matrix_gen(n):
    m = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            m[i][j] = random.random()
    return m

def vec_gen(n):
    v = np.zeros(n)
    for i in range(n):
        v[i] = random.random()
    return v

def prod_mat(A,B):
    n = len(A)
    C = np.zeros((n,n))
    if n == len(B):
        for i in range(n):
            for j in range(n):
                s=0
                for k in range(n):
                    s+= A[i][k]*B[k][j]
                C[i][j]=s
    return C
def complexity_graph():
    n = 50
    tab_x=range(1,n)
    tab_y_vec_opti=[]
    tab_y_vec= []
    tab_y_mat_opti=[]
    tab_y_mat=[]
    for j in tab_x:
        X = vec_gen(j)
        Y = vec_gen(j)
        V = vec_gen(j)
        M = matrix_gen(j)
        t1 = time.time()
        H = householder(X,Y)
        np.dot(H,V)
        t2 = time.time()
        householder_mul_vect_d(X,Y,V)
        t3 = time.time()
        H = householder(X,Y)
        prod_mat(H,M)
        t4 = time.time()
        householder_mul_mat_d(X,Y,M)
        t5 = time.time()
        tab_y_vec.append(t2-t1)
        tab_y_vec_opti.append(t3-t2)
        tab_y_mat.append(t4-t3)
        tab_y_mat_opti.append(t5-t4)
    return tab_x,tab_y_vec_opti,tab_y_vec,tab_y_mat_opti,tab_y_mat
'''
tab_x,tab_y_vec_opt,tab_y_vec,tab_y_mat_opti_tab_y_mat,tab_y_mat = complexity_graph()
plt.figure(1)
graph1 = plt.plot(tab_x,tab_y_vec,'b--',label='Non optimized')
graph2 = plt.plot(tab_x,tab_y_vec_opt,'r',label='Optimized')
plt.xlabel('n')
plt.ylabel('complexity')
plt.legend()


plt.figure(2)
graph3 = plt.plot(tab_x,tab_y_mat,'b--',label='Non optimized')
graph4 = plt.plot(tab_x,tab_y_mat_opti_tab_y_mat,'r',label='Optimized')
plt.xlabel('n')
plt.ylabel('complexity')
plt.legend()
plt.show()
'''

# x = np.array([[3],
#               [4],
#               [0]])
#
# y = np.array([[0],
#               [0],
#               [5]])
#
# h = householder(x, y)
#
# v = np.array([[2, 1, 0],
#               [0, 2, 1],
#               [1, 0, 0]])
#
# print np.dot(v, h)
# print householder_mul_mat_g(x, y, v)
