import configs
import os
import time
from functools import wraps
import numpy as np

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        w = open(os.path.join(configs.PATH_DIRECTORY_OUTPUT,f"{configs.NAME_FILE_TIME}.txt"), "a")
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        #print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        w.write(f"Function {func.__name__} Took {total_time:.4f} seconds\n")
        return result
    return timeit_wrapper

# This function created a dir with a name
def creationDir(nameDir:str):
    if(not(os.path.isdir(nameDir))):
        os.makedirs(nameDir)

# This function builds a diagonal positive Matrix and returns it
# @param nRow: the number of rows
# @param nColl: the number of columns
def diagonalM(nCols:int) -> np.ndarray:
        return np.diag(np.random.rand(1,nCols)[0])

def invSimpleDiag(D:np.ndarray):
    for i in range(D.shape[0]):
        D[i][i] = np.divide(1,D[i][i])
        # print(D)
        # input('premi')
    return D
# it's not optimized then it's better to use @ operator for matMul
# @timeit
# def productSimpleMatD(A:np.ndarray,D:np.ndarray) -> np.ndarray:
#     for j in range(A.shape[1]):
#         for i in range(A.shape[0]):
#             A[j][i] = A[j][i]*D[j][j]
#     return A

# Funzione che riceve come argomento le matrici e i vettori del problema e restituisce la matrice A e il vettore b
def instanceofMCF(D,E,b,c):
    A = np.block( [[D, E.T],[E, np.zeros((len(c),len(c)))]])
    b = np.hstack((b,c))
    return A,b

#Funzione che riceve la matrice di incidenza e la converte in un grafo di tipo networkx DA TESTARE
# def incidenceToGraph(A):
#     am =(np.dot(A,A.T)>0).astype(int)
    # print("Adjacence:",am)
    # G=nx.convert_matrix.from_numpy_array(am,parallel_edges=True,create_using=nx.DiGraph)
    # nx.draw(G)
    # plt.show()

def compute_x_cg(list_y:list[np.ndarray], D:np.ndarray, E_T:np.ndarray, b:np.ndarray)-> list[np.ndarray]:
    list_result:list[np.ndarray] = []
    D_inv:np.ndarray = np.linalg.inv(D)#invSimpleDiag(D)
    b = np.reshape(b,(len(b),1))
    invDb = np.dot(D_inv,b)
    invDEt = np.dot(D_inv,E_T)
    # print(f'shape D: {D.shape} shape y: {list_y[0].shape} shape b: {b.shape}, E_T: {E_T.shape}')
    x:np.ndarray
    v:np.ndarray
    for y in list_y:
        if(np.count_nonzero(y) > 0):
            x = - np.dot(invDEt,y) + invDb # -D^-1*E^T*y + D^-1*b
        else:
            x = invDb
        list_result.append(np.concatenate([x, y]))
        # list_result.append(x) # [x, y]
    return list_result

def compute_residual(list_x:list[np.ndarray],A:np.ndarray, b:np.ndarray)-> list[float]:
    list_result:list[float] = []
    b = np.reshape(b,(len(b),1))
    b_norm:float = np.linalg.norm(b)
    for x in list_x:
        # print(f'shape A: {A.shape} shape x: {x.shape} shape b: {b.shape}')
        # input('compute_residual')
        list_result.append(np.divide(np.linalg.norm(np.subtract(b, np.dot(A,x))),b_norm))
    return list_result

def compute_residual_reduced_system(list_y:list[np.ndarray],b_reduced:np.ndarray,A_reduced:np.ndarray):
    list_result:list[float] = []
    b = np.reshape(b_reduced, (len(b_reduced), 1))
    list_result.append(np.divide(np.linalg.norm(b), np.linalg.norm(b)))
    for y in list_y:
        # A^T*b - A^T*A*x = r
        # list_result.append(
        #     np.divide(np.linalg.norm(np.subtract(np.dot(A_reduced.T, b), np.dot(np.dot(A_reduced.T, A_reduced), y))), np.linalg.norm(np.dot(A_reduced.T, b)))
        # )
        list_result.append(
            np.divide(np.linalg.norm(np.subtract(b, np.dot(A_reduced, y))), np.linalg.norm(b))
        )
    return list_result

# https://tobydriscoll.net/fnc-julia/krylov/minrescg.html
def eig_limit(A:np.ndarray, num_iterations:int) -> list[float]:
    eig_values = np.linalg.eig(A)[0]
    # print(f'numero di autovalori: {len(eig_values)}')
    new_list_positive:list[float] = []
    new_list_negative:list[float] = []
    for eig in eig_values:
        if(eig > 1e-10):
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
        limit_convergence_MINRES.append(np.power(np.divide(np.sqrt(condition_number_positive*condition_number_negative) - 1, np.sqrt(condition_number_positive*condition_number_negative) + 1),np.floor((i + 1)/2))) 
    # * np.linalg.norm(r0)
    return limit_convergence_MINRES