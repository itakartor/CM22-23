import os
import configs
import time
import numpy as np
from util import timeit
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
    # è generale potrebbe essere resa più specifica per le posizioni non nulle
    n = len(b)
    x = np.zeros_like(b)
    
    if A.shape[1] == 1 :
        return b[0]/A[0,0]
    x[n-1] = b[n-1]/A[n-1, n-1]
    for i in range(n-2, -1, -1):
        bb = 0
        for j in range (i+1, n):
            bb = bb + A[i, j]*x[j]
        x[i] = (b[i] - bb)/A[i, i]
    return x

def QR_Householder(A,slower=False):
    m,n=A.shape
    Q = np.eye(m)
    R= A.copy()
    for j in (range(n)):
        u,_=householder_vector(R[j:,j])
        u.resize((len(u),1))
        H = np.eye(len(u)) - np.dot((2*u),u.T)
        if not slower:
            R[j:,j:]= R[j:,j:] - np.dot(2*u,np.dot(u.T,R[j:,j:]))
        else:
            R[j:,j:]= H  @ R[j:,j:]
        Q[:,j:]= Q[:,j:]@H    
    return Q,R

# @timeit
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
@timeit
def QR_givens_rotation(A:np.ndarray):
    """
    QR-decomposition of rectangular matrix A using the Givens rotation method.
    """
    # Initialization of the orthogonal matrix Q and the upper triangular matrix R
    n, m = A.shape
    Q:np.ndarray = np.eye(n) #Givens Matrix 1 ...1 c s -c -s
    R:np.ndarray = np.copy(A)
    # questa prende tutti gli indici della sotto triagonale sarebbe meglio prendere solo gli indici
    # di nostro interesse che coinvogono le 3 diagonali non nulle
    rows, cols = np.tril_indices(n, -1, m)   
    
    # sarebbe meglio non ricalcolare sempre Q ma calcolare solo l'ultima iterazione e aggiungerla a Q
    # la stessa cosa vale anche per R 
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
    c=np.dot(Q[:,:-1].T,b)
    x=back_substitution(R[:-1],c)
    return x

def __prodMV(A,v): 
    return np.matmul(A,v)


def lanczos(A,v1,v0,beta1,func=__prodMV):
    pk = func(A,v1)   #p = Av_j 
    
    alpha = np.dot(v1.T, pk) # alpha_j = vj.T Avj (Avj=pk) 
    
    pk = pk - alpha * v1 #w - alpha_j q_j
    
    v_next = pk - beta1 * v0 
    
    beta2 = np.linalg.norm(v_next) #  beta_j+1 = ||w_j||_2
    
    if not isclose(beta2,0.0):
        v_next = np.divide(v_next , beta2) #v_next normalization
    
    return alpha,beta2,v_next
    
# @timeit    
def custom_minres(A:np.ndarray, b:np.ndarray , m_dimension:int, x0:np.ndarray = None, tol:float = 1e-5, maxiter:int = None):
    #initialize variable
    start = time.time_ns()
    listTimeY:list[float] = []
    listYvectors:list[np.ndarray] = []
    listXvector:list[np.ndarray] = []
    n:int = len(b)
    b = np.reshape(b,(n,1))
    if maxiter == None:
        maxiter = n * 5
    
    exitmsgs = [
        "Exit Minres: A  does not define a symmetric matrix",
        "A solution to Ax = b was found at given tol at iteration {j}",
        "Lucky Breakdown solution of Ax=b with Beta_j=0 was found at iteration {j}",
        "Exit because max iteration reached iteration:{j}"
    ]
    exitRes = exitmsgs[1]
    
    #Check A density
    spar_ratio =__sparsity_ratio(A)
    if spar_ratio > 0.90:
        nnz = np.nonzero(A)
        print(f"A is a sparse matrix nonzero element of A = {len(nnz[0])} over {A.shape[0] * A.shape[1]} ")
    else:
        print("A is a dense matrix")
    
    #Check if A is symmetric
    if not np.allclose(A, A.T):
        return exitmsgs[0]
    
    #Check start point
    if x0 == None:
        w:np.ndarray = b.copy()
        w = w.reshape((n,1))
        #xc solution -> column vector  [x,y]
        xc:np.ndarray = np.zeros((n,1))
    else:
        w:np.ndarray = np.subtract(b,np.dot(A,x0))
        xc:np.ndarray = x0.copy()
    listYvectors.append(xc[m_dimension:,:])
    listXvector.append(xc)
    b_norm:float = np.linalg.norm(w) #Beta1 norm
    beta1:float = b_norm #beta1 -> betaj-1
    v1:np.ndarray = np.divide(w,beta1) # first vector of V
    v0:np.ndarray = np.zeros((len(v1),1))
    p1 = p0 = pk = np.zeros((len(v1),1)) #Vector of the matrix P_k= V_kR^-1 -> P_kR = V_k where R is a triagonal matrix of T_k QR factorization
    
    residual_list:list[float] = [b_norm] # residual of second solution
    T = np.zeros((0, 0)) #Initialize Empty Tridiagonal Matrix

    #Iteraions
    for j in range(maxiter):
        #Lanczos step iteration
        alpha,beta2,v_next = lanczos(A,v1,v0,beta1)
        #update of vj and vj+1
        v0 = v1.copy()
        v1 = v_next.copy()

        #Triagonal matrix T construction after Lanczos step
        T = tridiag(T,beta1,alpha,beta2)
        
        #if betaj+1 == 0 breakdown
        if isclose(beta2,0.0):
            exitRes = exitmsgs[2].format(j=j)
            break
        #update bj=bj+1 for the next iteration
        beta1 = beta2


        #QR factorization of T using givens rotations
        G,R = QR_givens_rotation(T)
        
        #G.T(beta_1 * e1)
        tj = G[0,j]*b_norm
        #Computation of the last 3  Pk vectors
        if j == 0 :
            pk = np.divide(v0,R[j,j])
            
        if j == 1 :
            pk = np.divide(np.subtract(v0 , np.dot(p1, R[j-1,j])),R[j,j])

        if j >= 2 :
            p01 = p1 * R[j-1,j] + p0 * R[j-2,j]
            diff = np.subtract(v0,p01)
            pk = np.divide(diff,R[j,j])

        #update 3 vectors for the next stepo    
        p0 = p1.copy()
        p1 = pk.copy()
        #increment of the solution xc
        xc = xc + np.dot(tj,pk)
        listYvectors.append(xc[m_dimension:,:])
        listXvector.append(xc)
        #Computation of the residual of the solution
        res = np.subtract(b,np.dot(A,xc))
        res = np.divide(np.linalg.norm(res),b_norm)
        residual_list.append(res)
        
        stop = time.time_ns()
        listTimeY.append((stop-start)/1e+6)

        #check if r is under the tollerance
        if res < tol:
            exitRes = exitmsgs[1].format(j=j)
            break
        
    #check if exit because maxiteration reached
    if j+1 == maxiter:
        exitRes = exitmsgs[3].format(j=j+1)
        
    # return j+1,x,xc,res,residual_list,exit
    w2 = open(os.path.join(configs.PATH_DIRECTORY_OUTPUT,f"{configs.NAME_FILE_STATISTIC_SOLUTION}.txt"), "a")
    # Algorithm& Graph& Rank& Iterations& Time execution& Residual
    w2.write(f"MINRES& & {maxiter - 1}& {j}& {listTimeY[-1]} ms& {res} \\ \n")
    w2.close()
    print(f"[MINRES] last iteration: {j}, residual: {res}, time: {listTimeY[-1]}ms, tollerance: {tol}")
    return j+1,xc,residual_list,exitRes,listTimeY,listYvectors,listXvector


#use lib to find eigenvalues of matix H
def eigenvalues(A):
    return np.linalg.eig(A)[0]

#print eigenvalut min and max given a matrix A
def min_max_eigenvalue(A):
    a = eigenvalues(A)
    return (a.min(),a.max())
    
    
#print the firt N eigenvalue
def print_first_N(a,N):
    try: acopy = a.copy()
    except: acopy = a[:]
    acopy.sort()
    max_value = min(N,len(a))
    for i in range(max_value):
        print('%1.5g\t' % acopy[i])
    print('')

