from IncidenceMatrix import IncidenceMatrix as im
from ConjugateGradient import ConjugateGradient as CG

listEMatrix:im=im()

listEMatrix = listEMatrix.buildIncidenceMatrix()
conjugate = CG(listEMatrix)
conjugate.start_CG()