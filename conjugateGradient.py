from incidenceMatrix import IncidenceMatrix
from util import *
import os
import numpy as np
import configs
import matplotlib as plt

class istanceMCF_CG:
    A:np.ndarray #E*D^-1*Et
    EMatrix:np.ndarray
    diagonalMatrix:np.ndarray
    vectorOfB:np.ndarray

class listOfPointsXY:
    listX:list[int]
    listY:list[int]
class PointXY:
    x:int
    y:int

# it's a list of the instances of the CG problems
class ConjugateGradient:
    listIstancesProblem:list[istanceMCF_CG] = []
    listofListPoints:list[listOfPointsXY] = []
    #initialize istance of the problems  with incidence matrixis 

    def __init__(self,eMatrixs:list[IncidenceMatrix]):
        self.listIstancesProblem:list = []
        self.listofPoints:list=[]
        # print(f"lunghezza {len(eMatrixs)}")
        for i in range(len(eMatrixs)):
            istanceMCF = istanceMCF_CG()
            c = self.buildArrayDeficit(eMatrixs[i].nodes)
            b = self.buildArrayCosts(eMatrixs[i].arches)
            istanceMCF.matrix = eMatrixs[i].m
            numOfArches:int = istanceMCF.matrix.shape[1]
            # this matrix is m*m that is arches number  
            istanceMCF.diagonalMatrix = diagonalM(numOfArches,numOfArches)
            
            #E*D^-1
            matrix_diagInv = istanceMCF.matrix @ invSimpleDiag(istanceMCF.diagonalMatrix)
            #E*D^-1*Et
            istanceMCF.A = matrix_diagInv @ istanceMCF.matrix.T
            #It's all values w
            istanceMCF.vectorOfb = (matrix_diagInv @ b ) - c
            self.listIstancesProblem.append(istanceMCF)

    def buildArrayDeficit(self,listNodes:list) -> np.array:
        c = np.array([])
        for node in listNodes:
            c = np.append(c,[[node.deficit]])
        return c
    
    def buildArrayCosts(self,dictArches:dict) -> np.array:
        b = np.array([])
        for key in dictArches.keys():
            b = np.append(b,[[dictArches[key].cost]])
        return b

    # algorithm
    # A is a matrix of system Ax = b
    # b is a vector of system Ax = b
    # x0 is the starting point
    # n is the number of iterations of the algorithm
    @timeit
    def getListPointCG(self,A:np.ndarray, b:np.ndarray, x:np.ndarray,path_output=configs.PATH_DIRECTORY_OUTPUT,solution_file=configs.NAME_FILE_SOLUTION,numIteration=100) ->listOfPointsXY:
        w = open(os.path.join(path_output,f"{solution_file}.txt"), "w")
        xGraph:list[int] = [] # number of iteration
        yGraph:list[int] = [] # difference between real b and artificial b
        listPoints:listOfPointsXY = listOfPointsXY()
        listPoints.listX = []
        listPoints.listY = []
        if(A.shape[1] != b.shape[0]):
            print('\n-------------------------------------')
            print("ERROR on dimension")
            print(f"dim A: {A.shape}, dim b: {b.shape}")
            print('-------------------------------------\n')
            return listPoints
        r:np.ndarray = np.reshape(np.copy(b),(4,1))# - A*x0 # residual Ax - b
        d:np.ndarray = r # directions vector
        alpha:float = 0
        beta:float = 0
        proveB:float
        for j in range(numIteration):
            Ad:np.ndarray = A @ d
            numAlpha:float = r.T @ r # this uses old r
            denAlpha:float = d.T @ Ad

            alpha = numAlpha/denAlpha
            x = x + alpha * d
            xGraph.append(j)
            
            proveB =  A @ x
            proveBNorm:float = np.linalg.norm(b - proveB)/np.linalg.norm(proveB) 
            
            yGraph.append(proveBNorm)
            r = r - alpha *Ad
            beta= r.T @ r / numAlpha # this uses new r
            d = r + beta * d
        listPoints.listX.extend(xGraph)
        listPoints.listY.extend(yGraph)
        w.write(f"A*x = b\n")
        w.write(f"Shape of A:{A.shape}\n A:\n{A}\n")
        w.write(f"Shape of x:{x.shape}\n CG x:\n{x}\n")
        w.write(f"Shape of b:{b.shape}\n CG b:\n{b}\n")
        w.write(f"Shape of proveB:{proveB.shape}\n CG proveB:\n{proveB}\n")
        w.write(f"CG proveBNorm:{proveBNorm}\n")
        
        w.close()
        return listPoints

    #compute the conjugate algorithm for all the problem instances
    @timeit
    def start_CG(self,draw_graph=configs.ACTIVE_DRAW_GRAPH, numIteration:int=100):
        i:int =1
        for instance in self.listIstancesProblem:
            points:listOfPointsXY = self.getListPointCG(
                A=instance.A,
                b=np.transpose(instance.vectorOfb),
                x=np.zeros((instance.A.shape[0],1)),
                numIteration=20
            )
            if draw_graph:
                plt.plot(points.listX,points.listY, label = f'iteration{i}')
                i += 1
            self.listofListPoints.append(points) 
        if draw_graph:
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.title("A simple line graph")
            # show a legend on the plot
            plt.legend()
            plt.show()
        return self.listofPoints

    #https://towardsdatascience.com/complete-step-by-step-conjugate-gradient-algorithm-from-scratch-202c07fb52a8