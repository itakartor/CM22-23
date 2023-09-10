from matplotlib.ticker import AutoLocator, IndexLocator, LinearLocator, LogLocator
import numpy as np
import IncidenceMatrixV2 
import util 
# from scipy.sparse.linalg import minresB
from MINRESV3 import  custom_minres
from conjugateGradient import ConjugateGradient as CG

import matplotlib.pyplot as plt

#bib reference
#https://web.stanford.edu/group/SOL/software/symmlq/PS75.pdf
#https://www.dmsa.unipd.it/~berga/Teaching/Phd/minres.pdf

#Build the Incidence Matrix of the graphs generated |complete graph ,grid graph,rmf graph|
listEMatrix:list[IncidenceMatrixV2.IncidenceMatrix] = IncidenceMatrixV2.buildIncidenceMatrix()

#for each Incdence matrix we build the Matrix A and the vector B from |D   E|=|c|
#                                                                     |E^t 0| |b|
v:int = 1
colors_list:list[str] = ['green','red','blue']
titles_list:list[str] = ['Complete','Grid','RMF']


fig,ax = plt.subplots(2, 1, constrained_layout = True)
for inM in listEMatrix:
    conjugate = CG(inM)
    res,last_iteration = conjugate.start_cg(inTol = 1e-4)
    
    E:np.ndarray = inM.m
    D:np.ndarray = util.diagonalM(E.shape[1])
    c:np.ndarray = [n.deficit for  n in inM.nodes]
    b:np.ndarray = [c.cost for c in inM.arcs]
    # print("Shapes:")
    # print("D shape:", D.shape, "E shape:", E.shape,"B len: ",len(b),"C len: ",len(c))
    A2,b2= util.instanceofMCF(D,E,b,c)
    # print("Result A and b shapes")
    # print("A:",A2.shape,"b",len(b2))
    # print("MinRes  MaxIter",len(b2)*5)
    
    j,xc,res2,exit = custom_minres(A2,b2,maxiter=A2.shape[0],tol = 1e-4)

    ax = plt.subplot(2,1,1)
    ax.grid(True)
    ax.set_title('Conjugate Gradient')
    ax.set_ylabel('Residual')
    ax.set_xlabel("Iteration")
    ax.set_yticks(np.arange(res[0], res[-1], 1e+1))
    ax.axline((conjugate.instanceProblem.A.shape[0] - 1, .0), (conjugate.instanceProblem.A.shape[0] - 1, .1), color='C3')
    ax.semilogy(res,linestyle="solid",color = 'red')
    ax.annotate(f"({last_iteration},{'{:.2e}'.format(res[-1])})", (last_iteration,res[-1]),xytext =(last_iteration - last_iteration/3,res[-1] + 5), arrowprops = dict(facecolor ='green',shrink = 0.05))
    
    ax = plt.subplot(2,1,2)
    ax.grid(True)
    ax.set_title('MINRES')
    ax.set_ylabel('Residual')
    ax.set_xlabel("Iteration")
    ax.set_yticks(np.arange(res2[0], res2[-1], 1e+1))
    ax.axline((A2.shape[0] - 1, .0), (A2.shape[0] - 1, .1), color='C3')
    ax.semilogy(res2,linestyle="solid",color = 'blue')
    ax.annotate(f"({j},{'{:.2e}'.format(res2[-1])})", (j,res2[-1]),xytext =(j - j/3,res2[-1] + 5), arrowprops = dict(facecolor ='green',shrink = 0.05))
    
    plt.suptitle(titles_list[v - 1],x = 0.55)
    v = v+1
    plt.show()
    plt.subplots_adjust(hspace=0.5)