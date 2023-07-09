import numpy as np
import util
from util import diagonalM, productSimpleMatD
nPow = 4
@util.timeit
def invS(D:np.ndarray):
    for i in range(pow(10,nPow)):
        D[i][i] = 1/D[i][i]
@util.timeit
def inv2(D:np.ndarray):
    I1 = np.linalg.inv(D)
    return I1
@util.timeit
def matMul(A:np.ndarray,D:np.ndarray):
    return A @ D


D:np.ndarray = diagonalM(pow(10,nPow))
#I2 = invS(D)
#I1 = inv2(D)
A:np.ndarray = np.random.rand(pow(10,nPow),pow(10,nPow))


# print(productSimpleMatD(A,D) - matMul(A,D))
