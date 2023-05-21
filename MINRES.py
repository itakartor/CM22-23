import os
import numpy as np
from incidenceMatrix import IncidenceMatrix

import configs
from util import diagonalM, listOfPointsXY, timeit


class istanceMCF_MINRES: # A * x = d
    # x^T = [x,y] d^T = [b,c]
    A:np.ndarray #[D E^T][x] = [b]
                 #[E  0][y] = [c]
    EMatrix:np.ndarray
    diagonalMatrix:np.ndarray
    vectorOfB:np.ndarray

class MINRES:
    listIstancesProblem:list[istanceMCF_MINRES] = []
    listofListPoints:list[listOfPointsXY] = []
    
    def __init__(self,eMatrices:list[IncidenceMatrix]):
        self.listIstancesProblem:list = []
        self.listofPoints:list=[]
        for i in range(len(eMatrices)):
            istanceMCF = istanceMCF_MINRES()
            istanceMCF.vectorOfb = self.buildDVector(eMatrices[i].nodes,eMatrices[i].arcs)
            istanceMCF.matrix = eMatrices[i].m
            numOfArches:int = istanceMCF.matrix.shape[1]
            # this matrix is m*m that is arches number  
            istanceMCF.diagonalMatrix = diagonalM(numOfArches,numOfArches)
            istanceMCF.A = np.block([
                    [istanceMCF.diagonalMatrix, istanceMCF.EMatrix.T],
                    [istanceMCF.EMatrix, np.zeros((len(eMatrices[i].nodes),len(eMatrices[i].nodes)))]
                ])
            self.listIstancesProblem.append(istanceMCF)
    
    def buildDVector(self,listNodes:list,dictArcs:dict):
        d = np.array([])
        for node in listNodes:
            d = np.append(d,[[node.deficit]])
        for key in dictArcs.keys():
            d = np.append(d,[[dictArcs[key].cost]])
        return d
    # algorithm
    # A is a matrix of system Ax = b
    # b is a vector of system Ax = b
    # x0 is the starting point
    # n is the number of iterations of the algorithm
    @timeit
    def getListPointMINRES(self,A:np.ndarray, b:np.ndarray, x0:np.ndarray,path_output=configs.PATH_DIRECTORY_OUTPUT,solution_file=configs.NAME_FILE_SOLUTION,numIteration=100) ->listOfPointsXY:
        w = open(os.path.join(path_output,f"{solution_file}.txt"), "w")
        xGraph:list[int] = [] # number of iteration
        yGraph:list[int] = [] # difference between real b and artificial b
        listPoints:listOfPointsXY = listOfPointsXY()
        listPoints.listX = []
        listPoints.listY = []
        if(A.shape[1] != b.shape[0]):
            w.write('\n-------------------------------------')
            w.write("ERROR on dimension")
            w.write(f"dim A: {A.shape}, dim b: {b.shape}")
            w.write('-------------------------------------\n')
            return listPoints
        x:np.ndarray = x0
        r:np.ndarray = np.reshape(np.copy(b - A @ x0),(4,1)) # - A*x0 # residual Ax - b
        d0:np.ndarray = r # directions vector
        s0:np.ndarray = A @ d0
        d1:np.ndarray = d0
        s1:np.ndarray = s0
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
