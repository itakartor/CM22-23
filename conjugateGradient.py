from incidenceMatrix import IncidenceMatrix

import os
import numpy as np
import configs
from matplotlib import pyplot as plt

from util import diagonalM, invSimpleDiag, listOfPointsXY, timeit

class istanceMCF_CG:
    A:np.ndarray #E*D^-1*Et
    EMatrix:np.ndarray
    diagonalMatrix:np.ndarray
    vectorOfB:np.ndarray
    name:str

# it's a list of the instances of the CG problems
class ConjugateGradient:
    listIstancesProblem:list[istanceMCF_CG] = []
    listofListPoints:list[listOfPointsXY] = []
    #initialize istance of the problems  with incidence matrixis 

    def __init__(self,eMatrices:list[IncidenceMatrix]):
        self.listIstancesProblem = []
        self.listofPoints = []
        # print(f"lunghezza {len(eMatrices)}")
        for i in range(len(eMatrices)):
            #print("I:",i)
            istanceMCF = istanceMCF_CG()
            istanceMCF.name=f"{eMatrices[i].generator.replace('./src/','')}-{i}"
            c = self.buildArrayDeficit(eMatrices[i].nodes)
            b = self.buildArrayCosts(eMatrices[i].arcs)
            istanceMCF.EMatrix = eMatrices[i].m
            numOfarcs:int = istanceMCF.EMatrix.shape[1]
            # this matrix is m*m that is arcs number  
            #print("this matrix is m*m that is arcs number")
            istanceMCF.diagonalMatrix = diagonalM(numOfarcs)
            
            #E*D^-1
            #print("D^-1")
            invDiag =invSimpleDiag(istanceMCF.diagonalMatrix)
            #print("E*D^-1")
            matrix_diagInv = istanceMCF.EMatrix @ invDiag
            #E*D^-1*Et
            #print("E*D^-1*Et") 
            istanceMCF.A = matrix_diagInv @ istanceMCF.EMatrix.T
            #It's all values w
            #print("It's all values w") 
            #print(eMatrices[i].generator,"MatrixDiagInv:",matrix_diagInv.shape,"B:",b.shape,"C:",c.shape)
            istanceMCF.vectorOfb = (matrix_diagInv @ b ) - c
            self.listIstancesProblem.append(istanceMCF)

    def buildArrayDeficit(self,listNodes:list) -> np.array:
        c = np.array([])
        for node in listNodes:
            c = np.append(c,[[node.deficit]])
        return c
    
    def buildArrayCosts(self,dictArcs:dict) -> np.array:
        b = np.array([])
        for key in dictArcs.keys():
            b = np.append(b,[[dictArcs[key].cost]])
        return b

    # algorithm
    # A is a matrix of system Ax = b
    # b is a vector of system Ax = b
    # x0 is the starting point
    # n is the number of iterations of the algorithm
    @timeit
    def getListPointCG(self,A:np.ndarray, b:np.ndarray, x0:np.ndarray,tol:float,path_output=configs.PATH_DIRECTORY_OUTPUT,solution_file=configs.NAME_FILE_SOLUTION_CG,numIteration=100,name="") ->listOfPointsXY:
        w = open(os.path.join(path_output,f"{solution_file}-{name}.txt"), "w")
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

            xGraph.append(j)
            
            r = r - alpha *Ad
            retTol = r.T @ r
            yGraph.append(retTol[0][0])
            if(retTol < tol*tol):
                break
            beta = r.T @ r / numAlpha # this uses new r
            d = r + beta * d
        listPoints.listX.extend(xGraph)
        listPoints.listY.extend(yGraph)
        w.write("A*x = b\n")
        w.write(f"Shape of A:{A.shape}\n A:\n{A}\n")
        w.write(f"Shape of x:{x.shape}\n CG x:\n{x}\n")
        w.write(f"Shape of b:{b.shape}\n CG b:\n{b}\n")
        w.write(f"last iteration: {last_iteration}, residual: {retTol[0][0]}")
        
        w.close()
        print(f"last iteration: {last_iteration}, residual: {retTol[0][0]}")

        return listPoints

    #compute the conjugate algorithm for all the problem instances
    @timeit
    def start_CG(self,draw_graph=configs.ACTIVE_DRAW_GRAPH, inNumIteration:int=0,inTol:float = 1e-5):
        i:int =1
        for instance in self.listIstancesProblem:
            print(f"rank matrix: {instance.A.shape[0]}")
            if(inNumIteration != 0):
                points:listOfPointsXY = self.getListPointCG(
                        A=instance.A,
                        b=np.transpose(instance.vectorOfb),
                        x0=np.zeros((instance.A.shape[0],1)),
                        tol=inTol,
                        numIteration=inNumIteration, # rank of the matrix
                        name=instance.name
                    )
            else:
                points:listOfPointsXY = self.getListPointCG(
                    A=instance.A,
                    b=np.transpose(instance.vectorOfb),
                    x0=None,
                    tol=inTol,
                    numIteration=instance.A.shape[0], # rank of the matrix
                    name=instance.name
                )
            if draw_graph:
                plt.subplot(3,1,i)
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