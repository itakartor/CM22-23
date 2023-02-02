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
        #result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        #print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        w.write(f"Function {func.__name__} Took {total_time:.4f} seconds\n")
        #return result
    return timeit_wrapper

# This function created a dir with a name
def creationDir(nameDir:str):
    if(not(os.path.isdir(nameDir))):
        os.makedirs(nameDir)

# This function builds a diagonal positive Matrix and returns it
# @param nRow: the number of rows
# @param nColl: the number of columns
def diagonalM(nRow:int, nCols:int) -> np.ndarray:
        return np.diag(np.random.rand(1,nCols)[0])

# this function has to calculate the condition Ex = c
@timeit
def testConservationRule(incidenceMatrix:np.ndarray,c:np.ndarray,x:np.ndarray = np.ndarray([])):
    # det = np.linalg.det(incidenceMatrix)
    # print("det: {det}")
    matrixInv = incidenceMatrix.getI()
    
    print(matrixInv)
    x = matrixInv @ c.T
    print(f"\nDIRECT TEST x: {x}\n")