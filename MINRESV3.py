import numpy as np
import time
from math import isclose, hypot
def __sparsity_ratio(X):
    return 1.0 - np.count_nonzero(X) / float(X.shape[0] * X.shape[1])
    
    
def householder_vector(x):
    s = np.linalg.norm(x)
    if x[0] >= 0:
        s = -s
    v = x.copy()
    v[0] = v[0] - s
    u = v / np.linalg.norm(v)
    return u,s

def back_substitution(A: np.ndarray, b: np.ndarray):
    n = len(b)
    x = np.zeros_like(b)
    
    if A.shape[1] == 1 :
        return b[0]/A[0,0]
    x[n-1] = b[n-1]/A[n-1, n-1]
    for i in range(n-2, -1, -1):
        bb = 0
        for j in range (i+1, n):
            bb += A[i, j]*x[j]
        x[i] = (b[i] - bb)/A[i, i]
    return x

def QR_Householder(A,slower=False):
    m,n=A.shape
    Q = np.eye(m)
    R= A.copy()
    for j in (range(n)):
        u,s=householder_vector(R[j:,j])
        u.resize((len(u),1))
        H = np.eye(len(u)) - np.dot((2*u),u.T)
        if not slower:
            R[j:,j:]= R[j:,j:] - np.dot(2*u,np.dot(u.T,R[j:,j:]))
        else:
            R[j:,j:]= H  @ R[j:,j:]
        Q[:,j:]= Q[:,j:]@H    
    return Q,R

def tridiag(T,beta1,alpha,beta2):
    if T.shape!=(0,0) : 
            zerorow=np.zeros((1,T.shape[1]))
            T=np.vstack(([T,zerorow]))
            newcol=np.zeros((T.shape[0],1))
            newcol[-3]=beta1
            newcol[-2]=alpha
            newcol[-1]=beta2
            T=np.hstack(([T,newcol]))
    else:
        T=np.vstack((alpha,beta2))
        T.reshape((2,1))
    return T

def QR_givens_rotation(A):
    """
    QR-decomposition of rectangular matrix A using the Givens rotation method.
    """
    # Initialization of the orthogonal matrix Q and the upper triangular matrix R
    n, m = A.shape
    Q = np.eye(n) #Givens Matrix 1 ...1 c s -c -s
    R = np.copy(A)

    rows, cols = np.tril_indices(n, -1, m)
    
    for (row, col) in zip(rows, cols):
        # If the subdiagonal element is nonzero, then compute the nonzero 
        # components of the rotation matrix        
        if R[row, col] != 0:
            #r = np.sqrt(R[col, col]**2 + R[row, col]**2)
            r = hypot(R[col, col],R[row, col])
            c, s = R[col, col]/r, -R[row, col]/r
            # The rotation matrix is highly discharged, so it makes no sense 
            # to calculate the total matrix product, we calculate G:[c -s] * R:[a b] [ca +(-s)e, cb+(-s)d]
            #                                                       [s  c]     [e,d]=[sa +ce,       sb+cd]
            R[col], R[row] = R[col]*c + R[row]*(-s), R[col]*s + R[row]*c
            # the same thing with the Givens matrix we made the product only for c,s Q= G_!.T*G_2.T Givens block transposed[c  s] 
            #                                                                                               [-s c]          
            Q[:, col], Q[:, row] = Q[:, col]*c + Q[:, row]*(-s), Q[:, col]*s + Q[:, row]*c
            
    return Q, R

def QR_ls(A: np.ndarray, b: np.ndarray):
    #Q,R=QR_Householder(A)
    Q,R=QR_givens_rotation(A)
    #Ax=b -> QRx=b-> Rx=Q.T*b
    #Rx=Q.T*b
    m,n=R.shape
    c=np.dot(Q[:,:-1].T,b)
    x=back_substitution(R[:-1],c)
    return x

def __prodMV(A,v): 
    return np.matmul(A,v)


def lanczos(A,v1,v0,beta1,func=__prodMV):
    pk= func(A,v1)   #p = Av_j 
    
    alpha = np.dot(v1, pk) # alpha_j = vj.T Avj (Avj=pk) 
    
    pk-= alpha * v1 #w - alpha_j q_j
    
    v_next = pk - beta1 * v0 
    
    beta2 = np.linalg.norm(v_next) #  beta_j+1 = ||w_j||_2
    
    if not isclose(beta2,0.0):
        v_next= np.divide(v_next , beta2) #v_next normalization
    
    return alpha,beta2,v_next
    
    
def minres(A, b , x0=None,tol=1e-5, maxiter=None, func=__prodMV):
    #initialize variable
    
    n = len(b)
    if maxiter == None:
        maxiter = n * 5
    
    exitmsgs = [
        "Exit Minres: A  does not define a symmetric matrix",
        "A solution to Ax = b was found at given tol at iteration {j}",
        "Lucky Breakdown solution of Ax=b with Beta_j=0 was found at iteration {j}"
        "Exit because max iteration reached it:{j}"
    ]
    exit = exitmsgs[1]
    
    #Check A density
    spar_ratio=__sparsity_ratio(A)
    if spar_ratio > 0.90:
        nnz=np.nonzero(A)
        print(f"A is a sparse matrix nonzero element of A = {len(nnz[0])} over {A.shape[0] * A.shape[1]} ")
    else:
        print("A is a dense matrix")
    
    #Check if A is symmetric
    if not np.allclose(A, A.T):
        return exitmsgs[0]
    
    #Check start point
    if x0 == None:
        w:np.ndarray = b.copy()
        #incomplete solution
        x:np.ndarray = np.zeros(len(b))
        #xc solution
        xc:np.ndarray = np.zeros(len(b))
    else:
        w:np.ndarray = np.subtract(np.reshape(b,(len(b),1)),np.dot(A,x0))
        x:np.ndarray=x0.copy()
        xc:np.ndarray = x0.copy()
    
    b_norm:float=np.linalg.norm(w) #Beta1 norm
    beta1:float  = b_norm #beta1 -> betaj-1
    v0:np.ndarray = 0
    v1:np.ndarray = w/beta1 # first vector of V
    V = np.array(v1.reshape((len(v1),1)))
    V = V.reshape((len(v1),1))
    p1 = p0 = pk =0 #Vector of the matrix P_k= V_kR^-1 -> P_kR = V_k where R is a triagonal matrix of T_k QR factorization
    res2:float=[] # residual of second solution
    res:float=[] #residual of firse solution
    T=np.zeros((0, 0)) #Initialize Empty Tridiagonal Matrix
    #Iteraions
    for j in range(maxiter):
        #Lanczos step iteration
        alpha,beta2,v_next=lanczos(A,v1,v0,beta1)
        #update of vj and vj+1
        v0=v1.copy()
        v1=v_next
        
        #Update of Martix V of Krylov space that we need to eliminate
        V=np.hstack((V,v1.reshape((len(v1),1))))

        #Triagonal matrix T construction after Lanczos step
        T=tridiag(T,beta1,alpha,beta2)
        
        #if betaj+1 == 0 breakdown
        if isclose(beta2,0.0):
            print(beta2)
            exit=exitmsgs[2].format(j=j)
            break
        #update bj=bj+1 for the next iteration
        beta1=beta2


        #QR factorization of T using givens rotations
        G,R=QR_givens_rotation(T)
        
        #G.T(beta_1 * e1)
        tj=G[0,j]*b_norm
        #Computation of the last 3  Pk vectors
        if j == 0 :
            pk = v0 / R[j,j]
        if j == 1 :
            pk= np.divide(v0 - p0 * R[j-1,j],R[j,j])
        if j>1 :
            p01=p1 * R[j-1,j] + p0 * R[j-2,j]
            diff=np.subtract(v0,p01)
            pk = np.divide(diff,R[j,j])
        #update 3 vectors for the next stepo    
        p0=p1
        p1=pk
        #increment of the solution xc
        xc +=  tj*pk
        
        #Computation of the residual of the solution
        r2 = np.subtract(np.reshape(b,(len(b),1)),np.dot(A,xc))
        r2 = np.linalg.norm(r2)
        res2.append(r2)
        
        #vector (Beta_1 e1) for the iteration j
        v = np.eye(j+2,1) * b_norm 
        #Working solution using V_k in memory for discover y_k = T_k-b1e1
        y = QR_ls(T,v) 
        
        #Computation of solution xk = Vk @ yk    
        x = np.dot(V[:,:-1],y)
        
        #Computation of the residual
        r = np.subtract(np.reshape(b,(len(b),1)),np.dot(A,x))
        r = np.linalg.norm(r)
        res.append(r)
        #check if r is under the tollerance
        
        if r < tol:
            exit=exitmsgs[1].format(j=j)
            break
    #check if exit because maxiteration reached
    if j==maxiter:
        exit=exit[3].format(j=j)
        
    return j+1,x,xc,res,res2,exit


#use lib to find eigenvalues of matix H
def eigenvalues(A):
    return np.linalg.eig(A)[0]

#print eigenvalut min and max given a matrix A
def min_max_eigenvalue(A):
    a=eigenvalues(A)
    return (a.min(),a.max())
    
    
#print the firt N eigenvalue
def print_first_N(a,N):
    try: acopy = a.copy()
    except: acopy = a[:]
    acopy.sort()
    max = min(N,len(a))
    for i in range(max):
        print('%1.5g\t' % acopy[i])
    print('')

