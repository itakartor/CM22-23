import sys

import numpy as np
from incidenceMatrix import IncidenceMatrix as im
from conjugateGradient import ConjugateGradient as CG
sys.path.insert(0, 'generators\\batch')


# listEMatrix:im=im()

# listEMatrix = listEMatrix.buildIncidenceMatrix()
# conjugate = CG(listEMatrix)
# conjugate.start_CG(numIteration=20)

# in this point we have to generate all incidences matrixes and
# generate all the solutions of the systems
n = 2
m = 3
print(np.block([
                    [np.ones((m, m)), np.ones((m, n))],
                    [np.ones((n, m)), np.zeros((n, n))]])
)