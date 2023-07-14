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

def lanczos(A,b,N, func=__prodMV):
    exit=N
    k=N
    Q = b.copy()/np.linalg.norm(b)
    alpha = np.zeros(k)
    beta = np.zeros(k+1)
    for m in range(k):
        w = func(A,Q[:,m])
        if m > 0 : 
            w-= beta[m] * Q[:,m-1]
        alpha[m] = np.dot(Q[:,m], w)
        w-=np.dot(alpha[m] , Q[:,m])
        beta[m+1]= np.linalg.norm(w)
        if np.abs(beta[m+1])<1e-10:
            exit=m
            print('Iter:',m)
            break 
        stack=np.divide(w.copy() , beta[m+1])
        stack=np.reshape(stack,(len(stack),1))
        Q = np.hstack((Q,stack))
    rbeta = beta[1:-1]
    H = np.diag(alpha)+ np.diag(rbeta, +1) + np.diag(rbeta,-1)
    return Q,H,exit




def ConstructMatrix(N):
    H = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            H[i, j] = float( 1+min(i, j) )
    print("H",H)
    return H

def eigenvalues(H):
    return np.linalg.eig(H)[0]

def print_first_last(a):
    print("Min  ",'%1.9g\t' % a.min())
    print("Max", '%1.9g' % a.max())

def print_first_N(a,N):
    try: acopy = a.copy()
    except: acopy = a[:]
    acopy.sort()
    max = min(N,len(a))
    for i in range(max):
        print('%1.5g\t' % acopy[i])
    print('')

def randomvector(N):
    return np.random.rand(N,1)            

def checkconvergence(N=5,A=None, b=None):
    
    #checks convergence of lanczos approximation to eigenvalues
    if A.all() == None:
        A = ConstructMatrix(N)
        A = A.T * A
        q = randomvector(N)
    else:
        q=np.reshape(b, (len(b), 1))
    True_eigvals = eigenvalues(A)

    print('True '),
    print_first_last(True_eigvals)
    Q,  h,exit = lanczos(A, q, A.shape[0])
    print('Q=', Q  )
    print('H=', h)
    for i in range(A.shape[0]-N,A.shape[0]+1):
         print('%i    ' % i)
         print_first_last(eigenvalues(h[:i,:i]))
         print_first_N( eigenvalues(h[:i,:i]) ,  N)
    print('Q=', Q  )
    print('H=', h)
    print(exit)
    print('eigenvalues via eig')
    print_first_N( True_eigvals ,  N)