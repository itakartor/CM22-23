import numpy as np
import IncidenceMatrixV2 
import util 
from scipy.sparse.linalg import minres
from MINRESV3 import  minres

import matplotlib.pyplot as plt

#bib reference
#https://web.stanford.edu/group/SOL/software/symmlq/PS75.pdf
#https://www.dmsa.unipd.it/~berga/Teaching/Phd/minres.pdf

#Build the Incidence Matrix of the graphs generated |complete graph ,grid graph,rmf graph|
listEMatrix = IncidenceMatrixV2.buildIncidenceMatrix()

#for each Incdence matrix we build the Matrix A and the vector B from |D   E|=|c|
#                                                                     |E^t 0| |b|
for inM in listEMatrix:
    E:np.ndarray = inM.m
    D:np.ndarray = util.diagonalM(E.shape[1])
    c:np.ndarray = [n.deficit for  n in inM.nodes]
    b:np.ndarray = [c.cost for c in inM.arcs]
    print("Shapes:")
    print("D shape:", D.shape, "E shape:", E.shape,"B len: ",len(b),"C len: ",len(c))
    A2,b2= util.instanceofMCF(D,E,b,c)
    print("Result A and b shapes")
    print("A:",A2.shape,"b",len(b2))
    print("MinRes  MaxIter",len(b2)*5)
    j,x,xc,r,r2,exit= minres(A2,b2,maxiter=116)
    res=r
    res2=r2
    """j,x,xc,r,r2,exit= minres(A2,b2,x0=xc,maxiter=30)
    res2+=r2
    j,x,xc,r,r2,exit= minres(A2,b2,x0=xc,maxiter=30)
    res2+=r2
    j,x,xc,rs,r2,exit= minres(A2,b2,x0=xc,maxiter=30)
    res2+=r2
    j,x,xc,rs,r2,exit= minres(A2,b2,x0=xc,maxiter=30)
    res2+=r2
    j,x,xc,rs,r2,exit= minres(A2,b2,x0=xc,maxiter=30)
    res2+=r2
    j,x,xc,rs,r2,exit= minres(A2,b2,x0=xc,maxiter=30)
    res2+=r2"""



    #k,res3=minresSlide(A2,b2,115)
    print("it:",j,"exit Minres:",exit)
    #print("res libreria",minres(A2,b2,show=True))
    #Plot of the residual
    
    fig,(ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('MinRes')
    
    ax1.semilogy(res,linestyle="dotted",color="green")
    ax1.set_ylabel('Residual')
    ax1.set_xlabel("Iteration")
    
    ax2.semilogy(res2,linestyle="solid",color="red")
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Residual')
    
    plt.show()
    break