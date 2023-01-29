from incidenceMatrix import IncidenceMatrix as im
import conjugateGradient as CG

eMatrix=im()
conjugate=CG([im])
conjugate.start_cg()