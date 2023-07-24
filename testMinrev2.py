import numpy as np
import IncidenceMatrixV2 
import util 
from scipy.sparse.linalg import minres
from MINRESV2 import minres2,checkconvergence,MINRES2, lanczos

#https://sci-hub.ru/https://doi.org/10.1137/9780898719987.ch7
#https://www.cs.cornell.edu/courses/cs6220/2017fa/CS6220_Lecture10.pdf

listEMatrix = IncidenceMatrixV2.buildIncidenceMatrix()
for inM in listEMatrix:
    E:np.ndarray = inM.m
    D:np.ndarray = util.diagonalM(E.shape[1])
    c:np.ndarray = [n.deficit for  n in inM.nodes]
    b:np.ndarray = [c.cost for c in inM.arcs]
    print("D shape:", D.shape, "E shape:", E.shape,"B len: ",len(b),"C len: ",len(c))
    A,b= util.instanceofMCF(D,E,b,c)
    print("A:",A.shape,"b",len(b))
    print("len",len(b)*5)
    print("Test Lanczos Algorithm")
    A = np.array([[3, 1, 2, 4], [1, 2, 1,3], [2, 1, 3, 1],[4, 3, 1,4]])
    b = np.array([1, 2, 3,4])
    Q,H,exit= lanczos(A,b,4)
    print("H?",Q.T@A@Q)
    print("Q",Q)
    print("H",H)
    print("exit",exit)
    #print(minres2(A,b))
    #checkconvergence(A=A,b=b)
    # break