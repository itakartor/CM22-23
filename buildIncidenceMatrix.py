
import os
from numpy import *
 



PATH_DIRECTORY_VALUE = ".\\testAnalyze"
NAME_FILE_MATRIX = "matrix"
class Arch():
    source:int
    destination:int
    maxCapacity:int
    cost:int
class IncidenceMatrix():
    m = array([])
    nRow:int = 0
    nCol:int = 0
    def __init__(self,nRow:int, nCol:int):
        self.nRow = nRow
        self.nCol = nCol
        #it's for init all zeros matrix
        self.m = zerosMatrix(nRow,nCol)
    def __str__(self) -> str:
        rStr = ''
        for i in range(self.nRow):
            rStr = rStr + '[ '
            for j in range(self.nCol):
                rStr = rStr + str(self.m[i][j]) + " "
            rStr = rStr + "] \n"
        return rStr

def zerosMatrix(nRow:int, nCol:int):
    return array([[0 for j in range(nCol)] for i in range(nRow)])            

def buildIncidenceMatrix() -> IncidenceMatrix:
    matrix = array([])
    for path in os.listdir(PATH_DIRECTORY_VALUE):
        w = open(os.path.join(PATH_DIRECTORY_VALUE, path + NAME_FILE_MATRIX), "w")
        r = open(os.path.join(PATH_DIRECTORY_VALUE, path), "r")
        nNodes = 0
        nArchs = 0
        for line in r:
            # if i found number of nodes
            match line[0]:
                case "c":
                    if("n" in line):
                        print(line.split("=")[1]) 
                        nNodes = int(line.split("=")[1])
                        break
                # in line with p there is number of nodes and archs
                # but it has to verify
                case "p":
                    arrValue = line.split(" ")
                    nArchs = int(arrValue[3])
                    nNodes = int(arrValue[2])
                    matrix:IncidenceMatrix = IncidenceMatrix(nNodes,nArchs)
                    break
                #case "a":
                #    arrValue = line.split(" ")    
                #    break
                #case "n":
                #    break
            
def main():
    print("MAIN")
    m = array([])
    # first param is the matrix/array
    # second param is the position of new column
    # third param is new value
    # fourth is the number of axis
    #m = insert(m, [1], [[1],[2],[3]], 1)
    #matrix:IncidenceMatrix = IncidenceMatrix(3,4);
    #print(str(matrix))
    #print(matix)

main()