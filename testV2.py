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
n_chart:int = len(listEMatrix)
v:int = 1
colors_list:list[str] = ['green','red','blue']
titles_list:list[str] = ['Complete','Grid','RMF']

fig,ax = plt.subplots(n_chart, 1, constrained_layout = True)
fig.suptitle('Conjugate Gradient',x = 0.55)
for inM in listEMatrix:
    conjugate = CG(inM)
    res = conjugate.start_cg()
    
    
    ax = plt.subplot(n_chart,1,v)
    ax.set_title(titles_list[v - 1])
    ax.set_ylabel('Residual')
    ax.set_xlabel("Iteration")
    ax.semilogy(res,linestyle="dotted",color = colors_list[v - 1])
    
    ax.axline((conjugate.instanceProblem.A.shape[0], .0), (conjugate.instanceProblem.A.shape[0], .1), color='C3')
    v = v+1
plt.show()
v = 1
fig,ax = plt.subplots(n_chart, 1, constrained_layout = True)
fig.suptitle('Minres',x = 0.55)
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
    # Ai = np.identity(5)
    # bi = np.ones(5)
    # for i in range(0,5):
    #     Ai[i,i] = Ai[i,i] + np.random.rand()   
    #     bi[i] = np.random.rand()   
    # j,x,xc,r,r2,exit= minres(Ai,bi,maxiter=5)
    # j,x,xc,r,r2,exit= minres(A2,b2,maxiter=A2.shape[0])
    
    j,xc,res2,exit = custom_minres(A2,b2,maxiter=A2.shape[0])
    

    ax = plt.subplot(n_chart,1,v)
    ax.set_title(titles_list[v - 1])
    ax.set_ylabel('Residual')
    ax.set_xlabel("Iteration")
    ax.semilogy(res2,linestyle="dotted",color = colors_list[v - 1])
    
    ax.axline((A2.shape[0], .0), (A2.shape[0], .1), color='C3')
    v = v+1

plt.show()