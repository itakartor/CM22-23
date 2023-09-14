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
TITLE_CHART_CG:str = 'Conjugate'
TITLE_CHART_MINRES:str = 'MINRES'
#Build the Incidence Matrix of the graphs generated |complete graph ,grid graph,rmf graph|
listEMatrix:list[IncidenceMatrixV2.IncidenceMatrix] = IncidenceMatrixV2.buildIncidenceMatrix()

#for each Incdence matrix we build the Matrix A and the vector B from |D   E|=|c|
#                                                                     |E^t 0| |b|
index_chart:int = 1
colors_list:list[str] = ['green','red','blue']
titles_list:list[str] = ['Complete','Grid','RMF']

tollerance:float = 1e-6

fig,ax = plt.subplots(2, 1, constrained_layout = True)
for inM in listEMatrix:
    print("Start solve: ",inM.generator)
    E:np.ndarray = inM.m
    D:np.ndarray = util.diagonalM(E.shape[1])
    c:np.ndarray = np.array([n.deficit for  n in inM.nodes])
    b:np.ndarray = np.array([c.cost for c in inM.arcs])
   
    A,b_full= util.instanceofMCF(D,E,b,c)

    conjugate = CG(inM)
    
    # Reduced System Conjugate gradient
    
    reduced_A = conjugate.instanceProblem.A
    reduced_b = conjugate.instanceProblem.b
    residual_Cgr,last_iteration_Cgr,listTimeY_Cgr,list_y_points = conjugate.start_cg(inTol = tollerance)

    # Reduced System MINRES

    # ax = plt.subplot(1,1,1)
    # ax.grid(True)
    # ax.set_title(TITLE_CHART_CG)
    # ax.set_ylabel(TITLE_COLUMN_RESIDUAL)
    # ax.set_xlabel(TITLE_COLUMN_ITERATION)
    # ax.set_yticks(np.arange(residual_Cgr[0], residual_Cgr[-1], 1e+1))
    # line = ax.axline((reduced_A.shape[0] - 1, .0), (reduced_A.shape[0] - 1, .1), color='C3')
    # ax.semilogy(residual_Cgr,linestyle="solid",color = 'red')
    
    # plt.suptitle(titles_list[index_chart - 1],x = 0.5)
    # plt.show()
    # plt.subplots_adjust(hspace=0.5)

    # # Original System MINRES

    last_iteration_MINRES,xc_MINRES,residual_MINRES,exit_MINRES,listTimeY2_MINRES,list_x_points_MINRES = custom_minres(A,b_full, D.shape[0],maxiter=A.shape[0],tol = tollerance)
 
    # ax = plt.subplot(1,1,1)
    # ax.grid(True)
    # ax.set_title(TITLE_CHART_MINRES)
    # ax.set_ylabel(TITLE_COLUMN_RESIDUAL)
    # ax.set_xlabel(TITLE_COLUMN_ITERATION)
    # ax.set_yticks(np.arange(residual_MINRES[0], residual_MINRES[-1], 1e+1))
    # ax.axline((A.shape[0] - 1, .0), (A.shape[0] - 1, .1), color='C3')
    # ax.semilogy(residual_MINRES,linestyle="solid",color = 'blue')

    # plt.suptitle(titles_list[index_chart - 1],x = 0.5)
    # plt.show()
    # plt.subplots_adjust(hspace=0.5)

    # #Time

    # ax = plt.subplot(2,1,1)
    # ax.grid(True)
    # ax.set_title(TITLE_CHART_CG)
    # ax.set_ylabel(TITLE_COLUMN_TIME)
    # ax.set_xlabel(TITLE_COLUMN_ITERATION)
    # ax.set_yticks(np.arange(listTimeY_Cgr[0], listTimeY_Cgr[-1], 2))
    # ax.semilogy(listTimeY_Cgr,linestyle="solid",color = 'red')

    # ax = plt.subplot(2,1,2)
    # ax.grid(True)
    # ax.set_title(TITLE_CHART_MINRES)
    # ax.set_ylabel(TITLE_COLUMN_TIME)
    # ax.set_xlabel(TITLE_COLUMN_ITERATION)
    # ax.set_yticks(np.arange(listTimeY2_MINRES[0], listTimeY2_MINRES[-1], 2))
    # ax.semilogy(listTimeY2_MINRES,linestyle="solid",color = 'blue')
 
    # plt.suptitle(titles_list[index_chart - 1],x = 0.5)
    # plt.show()
    # plt.subplots_adjust(hspace=0.5)

    # Compare x and residual on the Original System

    # list_x_points_cg = util.compute_x_cg(list_y_points, D, E.T, b)
    # list_residual_cg:list[float] = util.compute_residual(list_x_points_cg,A, b_full)
    # list_residual_MINRES:list[float] = util.compute_residual(list_x_points_MINRES,A, b_full)
    list_residual_cg:list[float] = util.compute_residual_reduced_system(list_y_points,D, E, b, c)
    list_residual_MINRES:list[float] = util.compute_residual_reduced_system(list_x_points_MINRES, D, E, b, c)
    
    ax = plt.subplot(2,1,1)
    ax.grid(True)
    ax.set_title(TITLE_CHART_CG)
    ax.set_ylabel(TITLE_COLUMN_RESIDUAL)
    ax.set_xlabel(TITLE_COLUMN_ITERATION)
    # ax.set_yticks(np.arange(list_residual_cg[0], list_residual_cg[-1], 1e+1))
    ax.semilogy(list_residual_cg,linestyle="solid",color = 'red')

    ax = plt.subplot(2,1,2)
    ax.grid(True)
    ax.set_title(TITLE_CHART_MINRES)
    ax.set_ylabel(TITLE_COLUMN_RESIDUAL)
    ax.set_xlabel(TITLE_COLUMN_ITERATION)
    # ax.set_yticks(np.arange(list_residual_MINRES[0], list_residual_MINRES[-1], 1e+1))
    ax.semilogy(list_residual_MINRES,linestyle="solid",color = 'blue')
    
    plt.suptitle(titles_list[index_chart - 1],x = 0.5)
    plt.show()
    plt.subplots_adjust(hspace=0.5)

    index_chart = index_chart+1