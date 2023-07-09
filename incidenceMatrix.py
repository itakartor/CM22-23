import os
import numpy as np
import configs
import util
import re
class Arc():
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
    def __init__(self,i="0",deficit=0) -> None:
        self.name=i
        self.deficit=deficit
    def __str__(self) -> str:
        return f"n Name: {self.name}, Deficit: {self.deficit}"

class IncidenceMatrix():
    m:np.ndarray = np.ndarray([], dtype='double')
    arcs:dict = {} #the key is the source-destination and the value is the Arc object 
    nodes:list = []
    nRow:int
    nCo:int
    archParallel:int
    maxCost:float
    minCost:float
    maxCapacity:float
    minCapacity:float
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
    # - A dictionary of arcs
    def __init__(self):
        self.m:np.ndarray = np.ndarray(None, dtype='double')
        self.arcs:dict = {}    #the key is the source-destination and the value is the Arc object 
        self.nodes:list = []
        self.nRow:int = -1
        self.nCol:int = -1
        self.archParallel:int = -1
        self.maxCost:float = float('-inf')
        self.minCost:float = float('inf')
        self.maxCapacity:float = float('-inf')
        self.minCapacity:float = float('inf')
        self.avgDeficit:int = -1
        self.totDeficit:int = -1
        self.seed:int = -1
        
    def fill_nodes(self, nRow, nodes):
        print("Numero Nodi:",nRow,"Nodi parsati:",len(nodes))
        for n in nodes:
            print(str(n))
        for indexn in range(1,nRow+1):
            if not any(node.name == str(indexn) for node in nodes):
                nodes.append(Node(i=indexn))
        return nodes
        
    def completeParser(self,w,r,nodes,arcs):
        print("Parsing complete graph start...")
        matrix:IncidenceMatrix = IncidenceMatrix()
        cIndex:int = 0
        for line in r:
            match line[0]:
                case "c":
                    if("generated" in line):
                        pieces = line.split(" ")
                        matrix.generator = f"{pieces[len(pieces) - 2]}"
                    elif(" parallel " in line):
                        matrix.archParallel = int(line[len(line)-2])
                    elif(" cost " in line):
                        matrix.maxCost = int(line[len(line) - 4])
                        matrix.minCost = int(line[len(line) - 2])
                    elif(" capacity " in line):
                        matrix.maxCapacity = int(line[len(line) - 4])
                        matrix.minCapacity = int(line[len(line) - 2])
                    elif("average deficit " in line):
                        matrix.avgDeficit = int(line[len(line)-2])
                    elif("total deficit " in line):
                        matrix.totDeficit = int(line[len(line)-2])
                    elif(" seed " in line):
                        matrix.seed = int(line[len(line)-2])        
                # in line with p there is number of nodes and arcs
                # but it has to verify
                case "p":
                    self.case_p(line,matrix)
                case "a":
                    arc=self.case_a(line,matrix,cIndex)
                    '''arcs[f'{arc.source}-{arc.destination}'] = arc'''
                    arcs[cIndex] = arc
                    cIndex = cIndex + 1
                case "n":
                    nodes.append(self.case_n(line))
        
        # print("++++++++++++++++++")
        # [print(nodes[i]) for i in range(len(nodes))]
        # print("++++++++++++++++++++++")
        # [print(arcs[i]) for i in range(len(arcs))]
        nodes=self.fill_nodes(matrix.nRow,nodes)
        matrix.nodes.extend(nodes)
        matrix.arcs.update(arcs)
        w.write(str(matrix))
        w.close()
        r.close()
        return matrix
    
    def case_p(self,line,matrix):
        arrValue = line.split()
        matrix.nCol = int(arrValue[3])
        matrix.nRow = int(arrValue[2])
        matrix.m = np.zeros((matrix.nRow, matrix.nCol))
        
    def case_a(self,line,matrix,cIndex):
        values = line.split()
        arc:Arc = Arc()
        arc.index = cIndex
        arc.source = int(values[1])
        arc.destination = int(values[2])
        matrix.m[arc.source - 1][arc.index] = 1
        matrix.m[arc.destination - 1][arc.index] = -1
        arc.maxCapacity = int(values[4])
        arc.cost = int(values[5])
        return arc
    
    def case_n(self,line):
        values = line.split()
        node:Node = Node()
        node.name = values[1]
        node.deficit = int(values[2])
        return node
    
    def extract_c_values(self,line,npar):
        values = re.findall(r'\b\d+\b', line)
        if len(values) == npar:
            return list(map(int, values))
        else:
            print(f"Errore: i valori {values} non possono essere estratti correttamente dalla stringa.")
            return

        
    def gridgraphParser(self,w,r,nodes,arcs):
        print("Parsing GRID GRAPH start...")
        matrix:IncidenceMatrix = IncidenceMatrix()
        matrix.generator="Grid Graph"
        cIndex:int = 0
        for line in r:
             match line[0]:
                case "c":
                    values=self.extract_c_values(line,5)
                    if (values != None):
                        print(f"Parametri C {values}")
                        matrix.minCapacity=values[0]
                        matrix.maxCapacity=values[1]
                        matrix.minCost=values[2]
                        matrix.maxCost=values[3]
                        matrix.seed = values[4]
                case "p":
                    self.case_p(line,matrix)
                case "a":
                    arc=self.case_a(line,matrix,cIndex)
                    '''arcs[f'{arc.source}-{arc.destination}'] = arc'''
                    arcs[cIndex] = arc
                    cIndex = cIndex + 1
                case "n":
                    nodes.append(self.case_n(line))
        matrix.arcs.update(arcs)
        nodes=self.fill_nodes(matrix.nRow,nodes)
        matrix.nodes.extend(nodes)
        w.write(str(matrix))
        w.close()
        r.close()
        return matrix
    
    
    
    def rmfParser(self,w,r,nodes,arcs):
        print("Parsing RMF graph start...")
        matrix:IncidenceMatrix = IncidenceMatrix()
        matrix.generator="RMF Graph"
        cIndex:int = 0
        for line in r:
             match line[0]:
                case "c":
                    values=self.extract_c_values(line,7)
                    if (values!= None):
                        print(f"Parametri C {values}")
                        matrix.minCapacity= values[2]
                        matrix.maxCapacity= values[3]
                        matrix.minCost= 0
                        matrix.maxCost= values[4]
                        matrix.seed = values[6]
                case "p":
                    self.case_p(line,matrix)
                case "n":
                    nodes.append(self.case_n(line))
                case "a":
                    arc=self.case_a(line,matrix,cIndex)
                    '''arcs[f'{arc.source}-{arc.destination}'] = arc'''
                    arcs[cIndex] = arc
                    cIndex = cIndex + 1

        matrix.arcs.update(arcs)
        nodes=self.fill_nodes(matrix.nRow,nodes)
        matrix.nodes.extend(nodes)
        
        w.write(str(matrix))
        w.close()
        r.close()
        return matrix
    
    def buildIncidenceMatrix( self,
            nodes:list=[],arcs:dict={},
            path_test=configs.PATH_DMX,path_output=configs.PATH_DIRECTORY_OUTPUT,
            matrix_file=configs.NAME_FILE_MATRIX_INCIDENCE
        ) ->list:
        retArrMatrix:list = []
        print(f"numFile: {len(os.listdir(path_test))}") 
        parser:dict={'com': self.completeParser , "ggr": self.gridgraphParser, "rmf": self.rmfParser}
        i:int = 0
        for path in os.listdir(path_test):
            util.creationDir(path_output)
            w = open(os.path.join(path_output,f"{matrix_file}{i}.txt"), "w")
            r = open(os.path.join(path_test, path), "r")
            retArrMatrix.append( parser[path[0:3]](w,r,nodes,arcs))
            nodes.clear()
            arcs.clear()
            i = i +1
        return retArrMatrix