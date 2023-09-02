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
    name:str

class MINRES:
    listIstancesProblem:list[istanceMCF_MINRES] = []
    listofListPoints:list[listOfPointsXY] = []
    
    def __init__(self,eMatrices:list[IncidenceMatrix]):
        self.listIstancesProblem:list = []
        self.listofPoints:list=[]
        for i in range(len(eMatrices)):
            istanceMCF = istanceMCF_MINRES()
            istanceMCF.name=f"{eMatrices[i].generator.replace('./src/','')}-{i}"
            istanceMCF.vectorOfb = self.buildDVector(eMatrices[i].nodes,eMatrices[i].arcs)
            istanceMCF.EMatrix = eMatrices[i].m
            numOfArches:int = istanceMCF.EMatrix.shape[1]
            # this matrix is m*m that is arches number  
            istanceMCF.diagonalMatrix = diagonalM(numOfArches)
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
    def getListPointMINRES(self,numIteration:int,tol:np.double,A:np.ndarray, b:np.ndarray, 
                           x0:np.ndarray,
                           path_output:str=configs.PATH_DIRECTORY_OUTPUT,
                           solution_file:str=configs.NAME_FILE_SOLUTION_MINRES,
                           name:str = '') ->listOfPointsXY:
        w = open(os.path.join(path_output,f"{solution_file}-{name}.txt"), "w")
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
        r:np.ndarray = b - A @ x0 # - A*x0 # residual Ax - b
        d0:np.ndarray = r # first directions vector
        q0:np.ndarray = A @ d0 # w nel nuovo algoritmo
        d1:np.ndarray = d0
        d2:np.ndarray
        q1:np.ndarray = q0
        q2:np.ndarray
        alpha:np.double = 0
        beta1:np.double = 0
        # proveB:np.double
        numAlpha:np.double
        denAlpha:np.double
        retTol:np.double
        last_iteration:int = 0
        
        for j in range(numIteration):
            try:
                last_iteration = j
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
                #proveB =  A @ x
                # update r
                r = r - alpha * q1
                retTol = r.T @ r
                yGraph.append(retTol[0][0])
                denBeta2 = q2.T @ q2
                if(retTol[0][0] < tol*tol or denAlpha == 0 or denBeta2 == 0):
                    print(f"last iteration: {j}, tollerance: {retTol[0][0]}")
                    break
                d0 = q1
                q0 = A @ q1
                beta1 = q0.T @ q1 / denAlpha # this uses new r
                d0 = d0 - beta1 * d1
                q0 = q0 - beta1 * q1    
                if(j >= 1):
                    beta2 = (q0.T @ q2) / (denBeta2)
                    d0 = d0 - beta2 * d2 
                    q0 = q0 - beta2 * q2
            except Exception:
                print(f'catched error {Exception} at iteration {j}')
                break
        listPoints.listX.extend(xGraph)
        listPoints.listY.extend(yGraph)
        w.write("A*x = b\n")
        w.write(f"Shape of A:{A.shape}\n A:\n{A}\n")
        w.write(f"Shape of x:{x.shape}\n CG x:\n{x}\n")
        w.write(f"Shape of b:{b.shape}\n CG b:\n{b}\n")
        # w.write(f"Shape of proveB:{proveB.shape}\n CG proveB:\n{proveB}\n")
        w.write(f"last iteration: {last_iteration}, tollerance: {retTol[0][0]}")
        
        w.close()
        print(f"last iteration: {last_iteration}, tollerance: {retTol[0][0]}")
        return listPoints

    #compute the conjugate algorithm for all the problem instances
    @timeit
    def start_MINRES(self,draw_graph=configs.ACTIVE_DRAW_GRAPH, numIteration:int=0, tol:np.double=1e-3):
        i:int =1
        for instance in self.listIstancesProblem:
            if(numIteration != 0):
                points:listOfPointsXY = self.getListPointMINRES(
                    numIteration,
                    tol,
                    A=instance.A,
                    b=np.transpose(instance.vectorOfb),
                    x0=np.zeros((instance.A.shape[0],1)),
                    name=instance.name
                )
            else:
                print(instance.A.shape[0]-1)
                points:listOfPointsXY = self.getListPointMINRES(
                    instance.A.shape[0]-1,
                    tol,
                    A=instance.A,
                    b=np.transpose(instance.vectorOfb),
                    x0=np.zeros((instance.A.shape[0],1)),
                    name=instance.name
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
