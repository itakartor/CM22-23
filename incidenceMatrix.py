import os
import numpy as np
import configs

class Arch():
    index:int
    source:int
    destination:int
    maxCapacity:int
    cost:int
    def __str__(self) -> str:
        return f"a Source: {self.source}, Destination: {self.destination}, MaxCapacity: {self.maxCapacity}, Cost: {self.cost}" 
class Node():
    name:str
    deficit:int
    def __str__(self) -> str:
        return f"n Name: {self.name}, Deficit: {self.deficit}"

class IncidenceMatrix():
    m:np.ndarray = np.ndarray([])
    arches:dict = {} #the key is the source-destination and the value is the Arch object 
    nodes:list = []
    nRow:int
    nColl:int
    archParallel:int
    maxCost:int
    minCost:int
    maxCapacity:int
    minCapacity:int
    avgDeficit:int
    totDeficit:int
    seed:int
    generator:str

    def __str__(self) -> str:
        rMatrixStr = str(self.m)
        rStr = f'''generated by the {self.generator} generator \n
                average parallel = {self.archParallel} \n
                max/min cost = {self.maxCost}/{self.minCost} \n
                max/min capacity = {self.maxCapacity}/{self.minCapacity} \n
                average deficit = {self.avgDeficit} \n
                total deficit = {self.totDeficit} \n
                seed = {self.seed} \n'''
        rStr = rStr + rMatrixStr
        return rStr
    # This fuction take 2 params:
    # - A list of Nodes 
    # - A dictionary of Arches
    def __init__(self,nodes:list=[],arches:dict={},path_test=configs.PATH_DIRECTORY_TEST,path_output=configs.PATH_DIRECTORY_OUTPUT,matrix_file=configs.NAME_FILE_MATRIX_INCIDENCE):
        cIndex:int = 0
        arrMatrix:list = []
        rMatrix:IncidenceMatrix = IncidenceMatrix()
        print(f"numFile: {len(os.listdir(path_test))}") 
        for path in os.listdir(path_test):
            
            i:int = 0
            creationDir(path_output)
            w = open(os.path.join(path_output,f"{matrix_file}{i}.txt"), "w")
            r = open(os.path.join(path_test, path), "r")
            for line in r:
                match line[0]:
                    case "c":
                        if("generated" in line):
                            pieces = line.split(" ")
                            rMatrix.generator = f"{pieces[len(pieces) - 2]}"
                        elif(" parallel " in line):
                            rMatrix.archParallel = int(line[len(line)-2])
                        elif(" cost " in line):
                            rMatrix.maxCost = int(line[len(line) - 4])
                            rMatrix.minCost = int(line[len(line) - 2])
                        elif(" capacity " in line):
                            rMatrix.maxCapacity = int(line[len(line) - 4])
                            rMatrix.minCapacity = int(line[len(line) - 2])
                        elif("average deficit " in line):
                            rMatrix.avgDeficit = int(line[len(line)-2])
                        elif("total deficit " in line):
                            rMatrix.totDeficit = int(line[len(line)-2])
                        elif(" seed " in line):
                            rMatrix.seed = int(line[len(line)-2])        
                    # in line with p there is number of nodes and arches
                    # but it has to verify
                    case "p":
                        arrValue = line.split(" ")
                        rMatrix.nColl = int(arrValue[3])
                        rMatrix.nRow = int(arrValue[2])
                        rMatrix.m = np.zeros((rMatrix.nRow, rMatrix.nColl))
                    case "a":
                        values = line.split()
                        arch:Arch = Arch()
                        arch.index = cIndex
                        cIndex = cIndex + 1
                        arch.source = int(values[1])
                        arch.destination = int(values[2])
                        
                        rMatrix.m[arch.source - 1][arch.index] = 1
                        rMatrix.m[arch.destination - 1][arch.index] = -1

                        arch.maxCapacity = int(values[4])
                        arch.cost = int(values[5])
                        arches[f'{arch.source}-{arch.destination}'] = arch
                    case "n":
                        values = line.split()
                        node:Node = Node()
                        node.name = values[1]
                        node.deficit = int(values[2])
                        nodes.append(node)
            w.write(str(rMatrix))
            i = i +1
            # print("++++++++++++++++++")
            # [print(nodes[i]) for i in range(len(nodes))]
            # print("++++++++++++++++++++++")
            # [print(arches[i]) for i in range(len(arches))]
            rMatrix.nodes.extend(nodes)
            rMatrix.arches.update(arches)
            arrMatrix.append(rMatrix)
            w.close()
            r.close()
        return arrMatrix 