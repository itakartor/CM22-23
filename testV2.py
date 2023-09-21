import os
import numpy as np
import IncidenceMatrixV2 
import util 
# from scipy.sparse.linalg import minresB
from MINRESV3 import  custom_minres
from conjugateGradient import ConjugateGradient as CG
import configs
import matplotlib.pyplot as plt

#bib reference
#https://web.stanford.edu/group/SOL/software/symmlq/PS75.pdf
#https://www.dmsa.unipd.it/~berga/Teaching/Phd/minres.pdf

TITLE_COLUMN_ITERATION:str = 'Iteration'
TITLE_COLUMN_RESIDUAL:str = 'Residual'
TITLE_COLUMN_TIME:str = 'Time (ms)'
TITLE_CHART_CG:str = 'Conjugate'
TITLE_CHART_MINRES:str = 'MINRES'

# https://tobydriscoll.net/fnc-julia/krylov/minrescg.html
def eig_limit(A:np.ndarray, num_iterations:int, r0:np.ndarray) -> list[float]:
    eig_values = np.linalg.eig(A)[0]
    # print(f'numero di autovalori: {len(eig_values)}')
    new_list_positive:list[float] = []
    new_list_negative:list[float] = []
    for eig in eig_values:
        if(eig > 1e-5):
            new_list_positive.append(eig)
        elif(eig < 0 and eig < -1e-10):
            new_list_negative.append(eig)
    # print(new_list_positive)
    # print('--------------------------------')
    # print(new_list_negative)
    max_eig_value_positive = np.max(new_list_positive)
    min_eig_value_positive = np.min(new_list_positive)
    max_eig_value_negative = np.min(new_list_negative)
    min_eig_value_negative = np.max(new_list_negative)
    # print(f'max P {max_eig_value_positive}, min P {min_eig_value_positive}, max N {max_eig_value_negative}, min N {min_eig_value_negative}')
    condition_number_positive:float = np.abs(max_eig_value_positive) / np.abs(min_eig_value_positive)
    condition_number_negative:float = np.abs(max_eig_value_negative) / np.abs(min_eig_value_negative)
    # print(f'CP {condition_number_positive}, CN {condition_number_negative}')
    # input('premi')
    limit_convergence_MINRES:list[float] = []
    for i in range(num_iterations):
        limit_convergence_MINRES.append(np.power(np.divide(np.sqrt(condition_number_positive*condition_number_negative) - 1, np.sqrt(condition_number_positive*condition_number_negative) + 1),np.floor((i + 1)/2))* np.linalg.norm(r0)) 
    # * np.linalg.norm(r0)
    return limit_convergence_MINRES
#Build the Incidence Matrix of the graphs generated |complete graph ,grid graph,rmf graph|
listEMatrix:list[IncidenceMatrixV2.IncidenceMatrix] = IncidenceMatrixV2.buildIncidenceMatrix()

#for each Incdence matrix we build the Matrix A and the vector B from |D   E|=|c|
#                                                                     |E^t 0| |b|
index_chart:int = 1
titles_list:list[str] = ['Complete','Grid','RMF']

tollerance:float = 1e-6
w2 = open(os.path.join(configs.PATH_DIRECTORY_OUTPUT,f"{configs.NAME_FILE_STATISTIC_SOLUTION}.txt"), "a")
w2.write("##############################################################################\n")
w2.close()
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

    ax = plt.subplot(1,1,1)
    ax.grid(True)
    ax.set_title(TITLE_CHART_CG)
    ax.set_ylabel(TITLE_COLUMN_RESIDUAL)
    ax.set_xlabel(TITLE_COLUMN_ITERATION)
    ax.set_yticks(np.arange(residual_Cgr[0], residual_Cgr[-1], 1e+1))
    line = ax.axline((reduced_A.shape[0] - 1, .0), (reduced_A.shape[0] - 1, .1), color='C3')
    ax.semilogy(residual_Cgr,linestyle="solid",color = 'red')
    
    plt.suptitle(titles_list[index_chart - 1],x = 0.55)
    plt.show()
    plt.subplots_adjust(hspace=0.5)

    # # Original System MINRES

    last_iteration_MINRES,xc_MINRES,residual_MINRES,exit_MINRES,listTimeY2_MINRES,list_x_points_MINRES = custom_minres(A,b_full, D.shape[0],maxiter=A.shape[0],tol = tollerance)
    list_limit_eig:list[float] = eig_limit(A,last_iteration_MINRES,b_full)
    ax = plt.subplot(1,1,1)
    ax.grid(True)
    ax.set_title(TITLE_CHART_MINRES)
    ax.set_ylabel(TITLE_COLUMN_RESIDUAL)
    ax.set_xlabel(TITLE_COLUMN_ITERATION)
    ax.set_yticks(np.arange(residual_MINRES[0], residual_MINRES[-1], 1e+1))
    ax.axline((A.shape[0] - 1, .0), (A.shape[0] - 1, .1), color='C3')
    ax.semilogy(list_limit_eig,linestyle="solid",color = 'green',label='Limit Minres')
    ax.semilogy(residual_MINRES,linestyle="solid",color = 'blue')

    plt.suptitle(titles_list[index_chart - 1],x = 0.55)
    plt.show()
    plt.subplots_adjust(hspace=0.5)

    # #Time

    ax = plt.subplot(2,1,1)
    ax.grid(True)
    ax.set_title(TITLE_CHART_CG)
    ax.set_ylabel(TITLE_COLUMN_TIME)
    ax.set_xlabel(TITLE_COLUMN_ITERATION)
    ax.set_yticks(np.arange(listTimeY_Cgr[0], listTimeY_Cgr[-1], 2))
    ax.semilogy(listTimeY_Cgr,linestyle="solid",color = 'red')

    ax = plt.subplot(2,1,2)
    ax.grid(True)
    ax.set_title(TITLE_CHART_MINRES)
    ax.set_ylabel(TITLE_COLUMN_TIME)
    ax.set_xlabel(TITLE_COLUMN_ITERATION)
    ax.set_yticks(np.arange(listTimeY2_MINRES[0], listTimeY2_MINRES[-1], 2))
    ax.semilogy(listTimeY2_MINRES,linestyle="solid",color = 'blue')
 
    plt.suptitle(f'Time Tracking {titles_list[index_chart - 1]}',x = 0.55)
    plt.show()
    plt.subplots_adjust(hspace=0.5)

    # Compare x and residual on the Original System

    list_residual_cg:list[float] = util.compute_residual_reduced_system(list_y_points, reduced_b, reduced_A)
    list_residual_MINRES:list[float] = util.compute_residual_reduced_system(list_x_points_MINRES, reduced_b, reduced_A)
    # list_limit_eig:list[float] = eig_limit(A,A.shape[0],list_residual_MINRES[0])

    ax = plt.subplot(1,1,1)
    ax.grid(True)
    ax.set_ylabel(TITLE_COLUMN_RESIDUAL)
    ax.set_xlabel(TITLE_COLUMN_ITERATION)
    ax.semilogy(list_residual_cg,linestyle="solid",color = 'red',label='CG')
    ax.semilogy(list_residual_MINRES,linestyle="solid",color = 'blue',label='MINRES')
    # ax.semilogy(list_limit_eig,linestyle="solid",color = 'green',label='Limit Minres')
    ax.legend()
    plt.suptitle(f'Residual Compare on Reduced System {titles_list[index_chart - 1]}',x = 0.5)
    plt.show()
    plt.subplots_adjust(hspace=0.5)

    index_chart = index_chart+1