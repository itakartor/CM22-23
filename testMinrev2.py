import numpy as np
import IncidenceMatrixV2 
import util 
from scipy.sparse.linalg import minres
from MINRESV2 import  lanczos_minres2, min_max_eigenvalue

import matplotlib.pyplot as plt

#https://sci-hub.ru/https://doi.org/10.1137/9780898719987.ch7
#https://www.cs.cornell.edu/courses/cs6220/2017fa/CS6220_Lecture10.pdf

listEMatrix = IncidenceMatrixV2.buildIncidenceMatrix()
for inM in listEMatrix:
    E:np.ndarray = inM.m
    D:np.ndarray = util.diagonalM(E.shape[1])
    c:np.ndarray = [n.deficit for  n in inM.nodes]
    b:np.ndarray = [c.cost for c in inM.arcs]
    print("D shape:", D.shape, "E shape:", E.shape,"B len: ",len(b),"C len: ",len(c))
    A2,b2= util.instanceofMCF(D,E,b,c)
    print("A:",A2.shape,"b",len(b2))
    print("MaxIter",len(b2)*5)
    #A = np.array([[3, 1, 2, 4], [1, 2, 1,3], [2, 1, 3, 1],[4, 3, 1,4]])
    #b = np.array([1, 2, 3,4])
    #K_A=np.linalg.cond(A2)
    #print("condA",K_A)
    #minA,maxA=min_max_eigenvalue(A2)
    #print("MINMAX",minA,maxA)
    #print("KA",np.abs(maxA)/np.abs(minA))
    j,x,xc,res,res2,exit= lanczos_minres2(A2,b2)
    #print("Residuto:",res,exit)
    #lib,exit=minres(A2,b2,show=True)
    print("it:",j,"exit Minres:",exit)
    print("Diff Residui ",res2[-1] - res[-1] )
    #print("res libreria",minres(A2,b2,show=True))
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('MinRes')
    
    ax1.semilogy(res,linestyle="dotted",color="green")
    ax1.set_ylabel('Residual')
    ax1.set_xlabel("Iteration")
    
    ax2.semilogy(res2,linestyle="solid",color="red")
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Residual')
    
    plt.show()
    break