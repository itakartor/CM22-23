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

TITLE_COLUMN_ITERATION:str = 'Iteration'
TITLE_COLUMN_RESIDUAL:str = 'Residual'
TITLE_COLUMN_TIME:str = 'Time (ms)'
TITLE_CHART_MINRESR:str = 'Reduced System MINRES'
TITLE_CHART_MINRES:str = 'Not Reduced System MINRES'
TITLE_CHART_CG:str = 'Not Reduced System Conjugate Gradient'
TITLE_CHART_CGR:str = 'Reduced System Conjugate Gradient'
#Build the Incidence Matrix of the graphs generated |complete graph ,grid graph,rmf graph|
listEMatrix:list[IncidenceMatrixV2.IncidenceMatrix] = IncidenceMatrixV2.buildIncidenceMatrix()

#for each Incdence matrix we build the Matrix A and the vector B from |D   E|=|c|
#                                                                     |E^t 0| |b|
index_chart:int = 1
colors_list:list[str] = ['green','red','blue']
titles_list:list[str] = ['Complete','Grid','RMF']

tollerance:float = 1e-4

fig,ax = plt.subplots(2, 1, constrained_layout = True)
for inM in listEMatrix:
    print("Start solve: ",inM.generator)
    E:np.ndarray = inM.m
    D:np.ndarray = util.diagonalM(E.shape[1])
    c:np.ndarray = [n.deficit for  n in inM.nodes]
    b:np.ndarray = [c.cost for c in inM.arcs]
    
    # print("Shapes:")
    # print("D shape:", D.shape, "E shape:", E.shape,"B len: ",len(b),"C len: ",len(c))
    
    # print("Result A and b shapes")
    # print("A:",A.shape,"b",len(b2))
    # print("MinRes  MaxIter",len(b2)*5)
   
    A,b_full= util.instanceofMCF(D,E,b,c)

    conjugate = CG(inM)
    
    # Reduced System Conjugate gradient
    
    reduced_A = conjugate.instanceProblem.A
    reduced_b = conjugate.instanceProblem.b
    residual_Cgr,last_iteration_Cgr,listTimeY_Cgr = conjugate.start_cg(inTol = tollerance)
        
    ax = plt.subplot(2,1,1)
    ax.grid(True)
    ax.set_title(TITLE_CHART_CGR)
    ax.set_ylabel(TITLE_COLUMN_RESIDUAL)
    ax.set_xlabel(TITLE_COLUMN_ITERATION)
    ax.set_yticks(np.arange(residual_Cgr[0], residual_Cgr[-1], 1e+1))
    line = ax.axline((reduced_A.shape[0] - 1, .0), (reduced_A.shape[0] - 1, .1), color='C3')
    line.set_label('pippo')
    ax.semilogy(residual_Cgr,linestyle="solid",color = 'red')
    ax.annotate(f"({last_iteration_Cgr} ; {'{:.2e}'.format(residual_Cgr[-1])})", (last_iteration_Cgr,residual_Cgr[-1]),xytext =(last_iteration_Cgr - last_iteration_Cgr/3,residual_Cgr[-1] + 5), arrowprops = dict(facecolor ='green',shrink = 0.05))
    
    # Reduced System MINRES

    last_iteration_MINRESR,xc_MINRESR,residual_MINRESR,exit_MINRESR,listTimeY2_MINRESR = custom_minres(reduced_A,reduced_b, maxiter=reduced_A.shape[0], tol = tollerance)

    ax = plt.subplot(2,1,2)
    ax.grid(True)
    ax.set_title(TITLE_CHART_MINRESR)
    ax.set_ylabel(TITLE_COLUMN_RESIDUAL)
    ax.set_xlabel(TITLE_COLUMN_ITERATION)
    ax.set_yticks(np.arange(residual_MINRESR[0], residual_MINRESR[-1], 1e+1))
    ax.axline((reduced_A.shape[0] - 1, .0), (reduced_A.shape[0] - 1, .1), color='C3')
    ax.semilogy(residual_MINRESR,linestyle="solid",color = 'blue')
    ax.annotate(f"({last_iteration_MINRESR} ; {'{:.2e}'.format(residual_MINRESR[-1])})", (last_iteration_MINRESR,residual_MINRESR[-1]),xytext =(last_iteration_MINRESR - last_iteration_MINRESR/3,residual_MINRESR[-1] + 5), arrowprops = dict(facecolor ='green',shrink = 0.05))
    
    plt.suptitle(titles_list[index_chart - 1],x = 0.55)
    plt.show()
    plt.subplots_adjust(hspace=0.5)


    ax = plt.subplot(2,1,1)
    ax.grid(True)
    ax.set_title(TITLE_CHART_CGR)
    ax.set_ylabel(TITLE_COLUMN_TIME)
    ax.set_xlabel(TITLE_COLUMN_ITERATION)
    ax.set_yticks(np.arange(listTimeY_Cgr[0], listTimeY_Cgr[-1], 2))
    ax.semilogy(listTimeY_Cgr,linestyle="solid",color = 'red')
    ax.annotate(f"({last_iteration_Cgr} ; {'{:.2f}'.format(listTimeY_Cgr[-1])})", (last_iteration_Cgr,listTimeY_Cgr[-1]),xytext =(last_iteration_Cgr - last_iteration_Cgr/3,listTimeY_Cgr[-1] - listTimeY_Cgr[-1]/2), arrowprops = dict(facecolor ='green',shrink = 0.05))
    
    ax = plt.subplot(2,1,2)
    ax.grid(True)
    ax.set_title(TITLE_CHART_MINRESR)
    ax.set_ylabel(TITLE_COLUMN_TIME)
    ax.set_xlabel(TITLE_COLUMN_ITERATION)
    ax.set_yticks(np.arange(listTimeY2_MINRESR[0], listTimeY2_MINRESR[-1], 2))
    ax.semilogy(listTimeY2_MINRESR,linestyle="solid",color = 'blue')
    ax.annotate(f"({last_iteration_MINRESR} ; {'{:.2f}'.format(listTimeY2_MINRESR[-1])})", (last_iteration_MINRESR,listTimeY2_MINRESR[-1]),xytext =(last_iteration_MINRESR - last_iteration_MINRESR/3,listTimeY2_MINRESR[-1] - listTimeY2_MINRESR[-1]/2), arrowprops = dict(facecolor ='green',shrink = 0.05))
    
    plt.suptitle(titles_list[index_chart - 1],x = 0.55)
    plt.show()
    plt.subplots_adjust(hspace=0.5)


    # Not Reduced System Conjugate

    conjugate.instanceProblem.A = A
    conjugate.instanceProblem.b = b_full
    residual_Cg,last_iteration_Cg,listTimeY_Cg = conjugate.start_cg(inTol = tollerance)
     
    ax = plt.subplot(2,1,1)
    ax.grid(True)
    ax.set_title(TITLE_CHART_CG)
    ax.set_ylabel(TITLE_COLUMN_RESIDUAL)
    ax.set_xlabel(TITLE_COLUMN_ITERATION)
    ax.set_yticks(np.arange(residual_Cg[0], residual_Cg[-1], 1e+1))
    ax.axline((A.shape[0] - 1, .0), (A.shape[0] - 1, .1), color='C3')
    ax.semilogy(residual_Cg,linestyle="solid",color = 'red')
    ax.annotate(f"({last_iteration_Cg} ; {'{:.2e}'.format(residual_Cg[-1])})", (last_iteration_Cg,residual_Cg[-1]),xytext =(last_iteration_Cg - last_iteration_Cg/3,residual_Cg[-1] + 5), arrowprops = dict(facecolor ='green',shrink = 0.05))
    
    # Not Reduced System MINRES

    last_iteration_MINRES,xc_MINRES,residual_MINRES,exit_MINRES,listTimeY2_MINRES = custom_minres(A,b_full, maxiter=A.shape[0], tol = tollerance)

    ax = plt.subplot(2,1,2)
    ax.grid(True)
    ax.set_title(TITLE_CHART_MINRES)
    ax.set_ylabel(TITLE_COLUMN_RESIDUAL)
    ax.set_xlabel(TITLE_COLUMN_ITERATION)
    ax.set_yticks(np.arange(residual_MINRES[0], residual_MINRES[-1], 1e+1))
    ax.axline((A.shape[0] - 1, .0), (A.shape[0] - 1, .1), color='C3')
    ax.semilogy(residual_MINRES,linestyle="solid",color = 'blue')
    ax.annotate(f"({last_iteration_MINRES} ; {'{:.2e}'.format(residual_MINRES[-1])})", (last_iteration_MINRES,residual_MINRES[-1]),xytext =(last_iteration_MINRES - last_iteration_MINRES/3,residual_MINRES[-1] + 5), arrowprops = dict(facecolor ='green',shrink = 0.05))
    
    plt.suptitle(titles_list[index_chart - 1],x = 0.55)
    plt.show()
    plt.subplots_adjust(hspace=0.5)

    ax = plt.subplot(2,1,1)
    ax.grid(True)
    ax.set_title(TITLE_CHART_CG)
    ax.set_ylabel(TITLE_COLUMN_TIME)
    ax.set_xlabel(TITLE_COLUMN_ITERATION)
    ax.set_yticks(np.arange(listTimeY_Cg[0], listTimeY_Cg[-1], 2))
    ax.semilogy(listTimeY_Cg,linestyle="solid",color = 'red')
    ax.annotate(f"({last_iteration_Cg} ; {'{:.2f}'.format(listTimeY_Cg[-1])})", (last_iteration_Cg,listTimeY_Cg[-1]),xytext =(last_iteration_Cg - last_iteration_Cg/3,listTimeY_Cg[-1] - listTimeY_Cg[-1]/2), arrowprops = dict(facecolor ='green',shrink = 0.05))
    
    ax = plt.subplot(2,1,2)
    ax.grid(True)
    ax.set_title(TITLE_CHART_MINRES)
    ax.set_ylabel(TITLE_COLUMN_TIME)
    ax.set_xlabel(TITLE_COLUMN_ITERATION)
    ax.set_yticks(np.arange(listTimeY2_MINRES[0], listTimeY2_MINRES[-1], 2))
    ax.semilogy(listTimeY2_MINRESR,linestyle="solid",color = 'blue')
    ax.annotate(f"({last_iteration_MINRESR} ; {'{:.2f}'.format(listTimeY2_MINRESR[-1])})", (last_iteration_MINRESR,listTimeY2_MINRESR[-1]),xytext =(last_iteration_MINRESR - last_iteration_MINRESR/3,listTimeY2_MINRESR[-1] - listTimeY2_MINRESR[-1]/2), arrowprops = dict(facecolor ='green',shrink = 0.05))
    
    plt.suptitle(titles_list[index_chart - 1],x = 0.55)
    index_chart = index_chart+1
    plt.show()
    plt.subplots_adjust(hspace=0.5)