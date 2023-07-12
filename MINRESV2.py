import numpy as np
from scipy.sparse import issparse
def __sparsity_ratio(X):
    return 1.0 - np.count_nonzero(X) / float(X.shape[0] * X.shape[1])

def minres2(A,b,x0=None,tol=1e-5, maxiter=None,plot=False):
    #initialize variable
    exitmsgs=["A  does not define a symmetric matrix","A solution to Ax = b was found, given tol"]
    n = len(b)
    if maxiter== None:
        maxiter = n * 5
    istop = 0
    itn = 0
    Anorm = 0
    rnorm = 0
    ynorm = 0
    done = False
    first = 'Enter minres.   '
    last = 'Exit  minres.   '

    print(first + 'Solution of symmetric Ax = b')
    print(first + f'n      =  {n}' )
    print(first + f'maxiter =  {maxiter}     rtol   =  %11.2e' % (tol))
    print()
    
  
    #Check A
    spar_ratio=__sparsity_ratio(A)
    if spar_ratio > 0.90:
        nnz=np.nonzero(A)
        print(f"A is a sparse matrix nonzero element of A = {len(nnz[0])} over {A.shape[0] * A.shape[1]} ")
    else:
        print("A is a dense matrix")
    
    #Check if A is symmetric
    if not np.allclose(A, A.T):
        return exitmsgs[0]
    
    
    if x0 == None:
        x0 = np.zeros((n,1))
        
    eps=np.finfo(float).eps
    
    #Setup y and v for the first Lanczos vectro v1
    # y  =  beta1 P' v1,  where  P = C**(-1).
    # v is really P' v1.
    b= np.reshape(b,(len(b),1))
    r = b - (A @ x0)
    p0 = r
    s0 = A @ p0
    p1 = p0
    s1 = s0
    for k in range(1,maxiter):
        p2 = p1
        p1 = p0
        s2 = s1
        s1 = s0
        alpha =  r.T @ s1 / s1.T @ s1
        x0 = x0 + alpha * p1
        r = r - alpha * s1
        if r.T@r < tol * tol:
            print ("tollerance")
            return x0,r
        p0 = s1
        s0 = A @ s1
        beta1 = s0.T @ s1 / (s1.T @s1)
        p0 = p0 - beta1 * p1
        s0 = s0 - beta1 * s1
        if k > 1:
            beta2 = s0.T * s2 / (s2.T @ s2)
            p0 = p0 - beta2 * p2
            s0 = s0 - beta2 * s2
    return x0,r