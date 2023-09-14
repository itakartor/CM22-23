import configs
import os
import time
from functools import wraps
import numpy as np
# import networkx as nx
import matplotlib.pyplot as plt

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

# this function has to calculate the condition Ex = c
@timeit
def testConservationRule(incidenceMatrix:np.ndarray,c:np.ndarray):
    # det = np.linalg.det(incidenceMatrix)
    # print("det: {det}")
    matrixInv = incidenceMatrix.getI()
    
    # print(matrixInv)
    x = matrixInv @ c.T
    # print(f"\nDIRECT TEST x: {x}\n")
    
def invSimpleDiag(D:np.ndarray):
    for i in range(D.shape[0]):
        D[i][i] = 1/D[i][i]
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
def incidenceToGraph(A):
    am =(np.dot(A,A.T)>0).astype(int)
    # print("Adjacence:",am)
    # G=nx.convert_matrix.from_numpy_array(am,parallel_edges=True,create_using=nx.DiGraph)
    # nx.draw(G)
    plt.show()

def compute_residual_reduced_system(list_y:list[np.ndarray],D:np.ndarray,E:np.ndarray,b_i:np.ndarray,c:np.ndarray):
    list_result:list[float] = []
    invDiag = invSimpleDiag(D)
    matrix_diagInv = np.matmul(E,invDiag)
    #E*D^-1*Et
    A:np.ndarray = np.matmul(matrix_diagInv,E.T)
    b:np.ndarray = np.subtract(np.matmul(matrix_diagInv,np.reshape(b_i,(b_i.shape[0],1))),np.reshape(c,(c.shape[0],1)))    
    for y in list_y:
        if(np.linalg.norm(y) > 0):
            y = y/np.linalg.norm(y)
        # print(f'mult: {np.matmul(A,y).shape}, b: {b.shape}, A:{A.shape}, y:{y.shape}')
        # print(np.subtract(np.reshape(b,(len(b),1)),np.dot(A,y)).shape)
        list_result.append(np.linalg.norm(np.subtract(np.reshape(b,(len(b),1)),np.dot(A,y))))
    return list_result

def compute_x_cg(list_y:list[np.ndarray], D:np.ndarray, E_T:np.ndarray, b:np.ndarray)-> list[np.ndarray]:
    list_result:list[np.ndarray] = []
    D_inv:np.ndarray = invSimpleDiag(D)
    
    for vector in list_y:
        v = np.matmul(np.matmul(-D_inv,E_T),vector) + np.matmul(D_inv,b).reshape((D_inv.shape[0],1))
        list_result.append(np.concatenate([v, vector])) # [x, y]
    return list_result

def compute_residual(list_x:list[np.ndarray],A:np.ndarray, b:np.ndarray)-> list[np.ndarray]:
    list_result:list[float] = []
    for x in list_x:
        # print(f'shape A: {A.shape} shape x: {x.shape} shape b: {b.shape}')
        list_result.append(np.linalg.norm(np.matmul(A,x) - b))
    return list_result