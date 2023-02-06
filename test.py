from incidenceMatrix import IncidenceMatrix as im
from conjugateGradient import ConjugateGradient as CG

listEMatrix:im=im()

listEMatrix = listEMatrix.buildIncidenceMatrix()
conjugate = CG(listEMatrix)
conjugate.start_CG(numIteration=20)