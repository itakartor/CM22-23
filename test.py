import sys

import numpy as np
from MINRES import MINRES
from incidenceMatrix import IncidenceMatrix as im
from conjugateGradient import ConjugateGradient as CG
sys.path.insert(0, 'generators\\batch')

# def err_callback():
#     break
# saved_handler = np.seterrcall(break)
save_err = np.seterr(over='raise')

listEMatrix:im=im()

listEMatrix = listEMatrix.buildIncidenceMatrix()
# conjugate = CG(listEMatrix)
# conjugate.start_CG()
minres = MINRES(listEMatrix)
minres.start_MINRES()

# in this point we have to generate all incidences matrixes and
# generate all the solutions of the systems
# n = 2
# m = 3
# print(np.block([
#                     [np.ones((m, m)), np.ones((m, n))],
#                     [np.ones((n, m)), np.zeros((n, n))]])
# )

# b = np.zeros((1,5))
# b = np.reshape(b,(5,1))
# print(b.shape[0])

#listEMatrix:im=im()

#listEMatrix = listEMatrix.buildIncidenceMatrix()
#minres = MINRES(listEMatrix)
#minres.start_MINRES(numIteration=1000,tol=1e-5)