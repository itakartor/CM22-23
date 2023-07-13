import numpy as np

def __sparsity_ratio(X):
    return 1.0 - np.count_nonzero(X) / float(X.shape[0] * X.shape[1])

def minres2(A,b,x0=None,tol=1e-5, maxiter=None,plot=False):
    #initialize variable
    
    n = len(b)
    if maxiter== None:
        maxiter = n * 5
        
    first = 'Enter minres.   '
    last = 'Exit  minres.   '

    print(first + 'Solution of symmetric Ax = b')
    print(first + f'n      =  {n}' )
    print(first + f'maxiter =  {maxiter}     rtol   =  %11.2e' % (tol))
    print()
    
    exitmsgs=[f"{last} A  does not define a symmetric matrix","A solution to Ax = b was found, given tol"]
  
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

def __prodMV(A,v):    
    return np.matmul(A,v)

def lanczos(A,b, func=__prodMV):
    k=len(b)
    Q =np.ndarray(b.copy/np.linalg.norm(b))
    alpha = np.zeros(k)
    beta = np.zeros(k+1)
    for m in range(k):
        w = func(A,Q[:,m])
        if m > 0 : w-= beta[m]* Q[:m-1]
        alpha[m] = (Q[:,m].T * w)[0,0]
        W-=alpha[m] * Q[:,m]
        beta[m+1]= np.linalg.norm(w)
        Q = np.hstack((Q,w.copy()/ beta[m+1]))
    rbeta= beta[1:-1]
    H = np.diag(alpha)+ np.diag(rbeta+1) + np.diag(rbeta,-1)
    return Q,H




def ConstructMatrix(N):
    H = np.ndarray(np.zeros([N,N]))
    for i in range(N):
        for j in range(N):
            H[i, j] = float( 1+min(i, j) )

def eigenvalues(H):
    return np.linalg.eig(H)[0]

def print_first_last(a):
    print('%1.9g\t' % a.min())
    print('%1.9g' % a.max())

def randomvector(N):
    v = np.random.random(N)
    n = np.sqrt( sum(v*v) )
    return np.ndarray(np.ndarray(v/n).T * 1L)            

def checkconvergence(N=10,N_to_display=5):
    #checks convergence of lanczos approximation to eigenvalues
    H = ConstructMatrix(N)
    H = H + 1j*H
    H = H.H * H
    True_eigvals = eigenvalues(H)

    print('True '),
    print_first_last(True_eigvals)
    

  #  v = [mat(zeros(N)).T, randomvector(N)]
    v = randomvector(N)
    #V, h = arnoldi(H, v, N)
    #V, h  = lanczos(H, v, N)
    V,  h = method(H, v, N)
 #   print 'V=',  V
    for i in range(1,N+1):
        print '%i    ' % i,
        #print_first_last(eigenvalues(h[:i,:i]))
        print_first_N( eigenvalues(h[:i,:i]) ,  i)
    print 'eigenvalues via eig(flapack)'
    print_first_N( True_eigvals ,  N)