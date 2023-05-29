import os
from matplotlib import pyplot as plt
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
        print(len(eMatrices))
        self.listIstancesProblem:list = []
        self.listofPoints:list=[]
        for i in range(len(eMatrices)):
            istanceMCF = istanceMCF_MINRES()
            istanceMCF.vectorOfb = self.buildDVector(eMatrices[i].nodes,eMatrices[i].arcs)
            istanceMCF.EMatrix = eMatrices[i].m
            numOfArches:int = istanceMCF.EMatrix.shape[1]
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
        d = np.reshape(d,(1,d.shape[0]))
        return d
    
    # algorithm
    # A is a matrix of system Ax = b
    # b is a vector of system Ax = b
    # x0 is the starting point
    # n is the number of iterations of the algorithm
    # tol is the tollerance for break the cycle of the algorithm
    @timeit
    def getListPointMINRES(self,A:np.ndarray, b:np.ndarray, 
                           x0:np.ndarray,tol:float,
                           path_output:str=configs.PATH_DIRECTORY_OUTPUT,
                           solution_file:str=configs.NAME_FILE_SOLUTION_MINRES,
                           numIteration:int=100) ->listOfPointsXY:
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
        print(f"A: {A.shape}")
        print(f"x0: {x0.shape}")
        print(f"b: {b.shape}")
        r:np.ndarray = b - A @ x0 # - A*x0 # residual Ax - b
        print(f"r: {r.shape}")
        d0:np.ndarray = r # first directions vector
        q0:np.ndarray = A @ d0
        d1:np.ndarray = d0
        d2:np.ndarray
        q1:np.ndarray = q0
        q2:np.ndarray
        alpha:float = 0
        beta1:float = 0
        proveB:float
        numAlpha:float
        denAlpha:float
        retTol:float
        for j in range(numIteration):
            d2 = d1
            d1 = d0
            q2 = q1
            q1 = q0
            numAlpha = r.T @ q1
            denAlpha = q1.T @ q1
            alpha = numAlpha/denAlpha
            # update x
            x = x + alpha * d1
            # for the point of a feasible graph
            xGraph.append(j)
            proveB =  A @ x
            proveBNorm:float = np.linalg.norm(b - proveB)/np.linalg.norm(proveB) 
            yGraph.append(proveBNorm)
            # update r
            r = r - alpha * q1
            print(f'{r.shape[0]}, {r.shape[1]}')
            retTol = r.T @ r
            #print(retTol)
            if(retTol < tol*tol):
                break
            d0 = q1
            q0 = A @ q1
            beta1 = q0.T @ q1 / denAlpha # this uses new r
            d0 = d0 - beta1 * d1
            q0 = q0 - beta1 * q1    
            if(j >= 1):
                beta2 = (q0.T @ q2) / (q2.T @ q2)
                d0 = d0 - beta2 * d2 
                q0 = q0 - beta2 * q2 
        listPoints.listX.extend(xGraph)
        listPoints.listY.extend(yGraph)
        w.write("A*x = b\n")
        w.write(f"Shape of A:{A.shape}\n A:\n{A}\n")
        w.write(f"Shape of x:{x.shape}\n CG x:\n{x}\n")
        w.write(f"Shape of b:{b.shape}\n CG b:\n{b}\n")
        w.write(f"Shape of proveB:{proveB.shape}\n CG proveB:\n{proveB}\n")
        w.write(f"CG proveBNorm:{proveBNorm}\n")
        
        w.close()
        return listPoints

    #compute the conjugate algorithm for all the problem instances
    @timeit
    def start_MINRES(self,draw_graph=configs.ACTIVE_DRAW_GRAPH, numIteration:int=100):
        i:int =1
        for instance in self.listIstancesProblem:
            points:listOfPointsXY = self.getListPointMINRES(
                A=instance.A,
                b=np.transpose(instance.vectorOfb),
                x0=np.zeros((instance.A.shape[0],1)),
                tol=0.000001,
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
