import time
from IncidenceMatrixV2 import IncidenceMatrix

import os
import numpy as np
import configs
from matplotlib import pyplot as plt

from util import diagonalM, invSimpleDiag, timeit

class istance_cg:
    A:np.ndarray #E*D^-1*Et
    EMatrix:np.ndarray
    diagonalMatrix:np.ndarray
    vectorOfB:np.ndarray
    name:str

# it's a list of the instances of the CG problems
class ConjugateGradient:
    instanceProblem:istance_cg
    #initialize istance of the problems  with incidence matrixis 

    def __init__(self,eMatrix:IncidenceMatrix):
        # print(f"lunghezza {len(eMatrices)}")
        #print("I:",i)
        self.instanceProblem = istance_cg()
        self.instanceProblem.name=f"{eMatrix.generator.replace('./src/','')}"
        c = self.build_array_deficit(eMatrix.nodes)
        b = self.build_array_costs(eMatrix.arcs)
        self.instanceProblem.EMatrix = eMatrix.m
        numOfarcs:int = self.instanceProblem.EMatrix.shape[1]
        # this matrix is m*m that is arcs number  
        #print("this matrix is m*m that is arcs number")
        self.instanceProblem.diagonalMatrix = diagonalM(numOfarcs)
        
        #E*D^-1
        #print("D^-1")
        invDiag = invSimpleDiag(self.instanceProblem.diagonalMatrix)
        #print("E*D^-1")
        matrix_diagInv = self.instanceProblem.EMatrix @ invDiag
        #E*D^-1*Et
        #print("E*D^-1*Et") 
        self.instanceProblem.A = matrix_diagInv @ self.instanceProblem.EMatrix.T
        #It's all values w
        #print("It's all values w") 
        #print(eMatrix.generator,"MatrixDiagInv:",matrix_diagInv.shape,"B:",b.shape,"C:",c.shape)
        self.instanceProblem.vectorOfb = (matrix_diagInv @ b ) - c

    def build_array_deficit(self,listNodes:list) -> np.array:
        c = np.array([])
        for node in listNodes:
            c = np.append(c,[[node.deficit]])
        return c
    
    def build_array_costs(self,listArcs:dict) -> np.array:
        b = np.array([])
        for arc in listArcs:
            b = np.append(b,[[arc.cost]])
        return b

    # algorithm
    # A is a matrix of system Ax = b
    # b is a vector of system Ax = b
    # x0 is the starting point
    # n is the number of iterations of the algorithm
    def get_list_point_cg(self,A:np.ndarray, b:np.ndarray, x0:np.ndarray,tol:float,path_output=configs.PATH_DIRECTORY_OUTPUT,solution_file=configs.NAME_FILE_SOLUTION_CG,numIteration=100,name="") -> tuple[list[float],int,list[float]]:
        start = time.time_ns()
        w = open(os.path.join(path_output,f"{solution_file}-{name}.txt"), "w")
        listPointsY:list[float] = []
        listTimeY:list[float] = []
        if(A.shape[1] != b.shape[0]):
            print('\n-------------------------------------')
            print("ERROR on dimension")
            print(f"dim A: {A.shape}, dim b: {b.shape}")
            print('-------------------------------------\n')
            return listPointsY
        r:np.ndarray
        if(x0 == None):
            x = np.zeros((A.shape[0],1))
            r = np.reshape(np.copy(b),(A.shape[1],1))# - A*x0 # residual Ax - b
        else:
            x = x0.copy()
            r = np.reshape(np.subtract(np.copy(b), A @ x0),(A.shape[1],1))# - A*x0 # residual Ax - b
        d:np.ndarray = r # directions vector
        alpha:float = 0
        beta:float = 0
        last_iteration:int = 0
        for j in range(numIteration):
            last_iteration = j
            Ad:np.ndarray = A @ d
            numAlpha:float = r.T @ r # this uses old r
            denAlpha:float = d.T @ Ad

            alpha = numAlpha/denAlpha
            x = x + alpha * d
            
            r = r - alpha *Ad
            retTol = r.T @ r
            listPointsY.append(retTol[0][0])
            if(retTol < tol*tol):
                break
            beta = r.T @ r / numAlpha # this uses new r
            d = r + beta * d
            #do some stuff
            stop = time.time_ns()
            listTimeY.append((stop-start)/1e+6)
        w.write("A*x = b\n")
        w.write(f"Shape of A:{A.shape}\n A:\n{A}\n")
        w.write(f"Shape of x:{x.shape}\n CG x:\n{x}\n")
        w.write(f"Shape of b:{b.shape}\n CG b:\n{b}\n")
        w.write(f"[CG] last iteration: {last_iteration}, residual: {retTol[0][0]}, time: {listTimeY[-1]}ms")
        
        w.close()
        print(f"[CG] last iteration: {last_iteration}, residual: {retTol[0][0]}, time: {listTimeY[-1]}ms")

        # return listPoints
        return listPointsY,last_iteration,listTimeY

    #compute the conjugate algorithm for all the problem instances
    @timeit
    def start_cg(self, inNumIteration:int=0,inTol:float = 1e-5):
        print(f"rank matrix: {self.instanceProblem.A.shape[0]}")
        points:list[float] = []
        if(inNumIteration != 0):
            points,last_iteration,listTimeY = self.get_list_point_cg(
                    A=self.instanceProblem.A,
                    b=np.transpose(self.instanceProblem.vectorOfb),
                    x0=np.zeros((self.instanceProblem.A.shape[0],1)),
                    tol=inTol,
                    numIteration=inNumIteration, # rank of the matrix
                    name=self.instanceProblem.name
                )
        else:
            points,last_iteration,listTimeY = self.get_list_point_cg(
                    A=self.instanceProblem.A,
                    b=np.transpose(self.instanceProblem.vectorOfb),
                    x0=None,
                    tol=inTol,
                    numIteration=self.instanceProblem.A.shape[0], # rank of the matrix
                    name=self.instanceProblem.name
                )
        return points,last_iteration,listTimeY