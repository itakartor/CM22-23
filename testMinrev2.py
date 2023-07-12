import numpy as np
import IncidenceMatrixV2 
import util 
from scipy.sparse.linalg import minres
from MINRESV2 import minres2

listEMatrix = IncidenceMatrixV2.buildIncidenceMatrix()
for inM in listEMatrix:
    E = inM.m
    D = util.diagonalM(E.shape[1])
    c = [n.deficit for  n in inM.nodes]
    b = [c.cost for c in inM.arcs]
    print("D shape:", D.shape, "E shape:", E.shape,"B len: ",len(b),"C len: ",len(c))
    A,b= util.instanceofMCF(D,E,b,c)
    print("A:",A.shape,"b",len(b))
    #x,exitcode=minres(A,b,show=True)
    print("Exit msg: ",minres2(A,b))