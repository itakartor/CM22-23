import sys
from incidenceMatrix import IncidenceMatrix as im
from conjugateGradient import ConjugateGradient as CG
sys.path.insert(0, 'generators\\batch')
from generateALL import generateAllGraphs



listEMatrix:im=im()

listEMatrix = listEMatrix.buildIncidenceMatrix()
conjugate = CG(listEMatrix)
conjugate.start_CG(numIteration=20)

# in this point we have to generate all incidences matrixes and
# generate all the solutions of the systems