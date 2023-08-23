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
    C = np.zeros((n,n))
    for i in range(n-2, -1, -1):
        bb = 0
        for j in range (i+1, n):
            bb += A[i, j]*x[j]
        C[i, i] = b[i] - bb
        x[i] = C[i, i]/A[i, i]
    
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

def givens_rotation(A):
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
            
    return Q[:, :m], R[:m]

def QR_ls(A: np.ndarray, b: np.ndarray):
    #Q,R=QR_Householder(A)
    Q,R=givens_rotation(A)
    #Ax=b -> QRx=b-> Rx=Q.T*b
    #Rx=Q.T*b
    m,n=R.shape
    c=np.dot(Q[:,:n].T,b)
    x=back_substitution(R[:n,:n],c)
    
    return x

def __prodMV(A,v): 
    return np.matmul(A,v)


def lanczos_minres2(A, b , x0=None,tol=1e-5, maxiter=None, func=__prodMV):
    #initialize variable
    n = len(b)

    if maxiter== None:
        maxiter = n * 5
    
    exitmsgs = [
        "Exit Minres: A  does not define a symmetric matrix",
        "A solution to Ax = b was found at given tol at iteration {j}",
        "Lucky Breakdown solution of Ax=b with Beta_j=0 was found at iteration {j}"
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
        w = b.copy()
    else:
        w = np.subtract(np.reshape(b,(len(b),1)),np.dot(A,x0))
    b_norm=np.linalg.norm(b)
    beta1=np.linalg.norm(w)
    Q = np.zeros((len(b),maxiter+1))
    Q[:,0] = w/beta1 #this can be considered as Beta_1 np.linalg.norm(b) and b as w_1
    T = np.zeros((maxiter+1,maxiter)) #Triadiagonal Matrix
    alpha = np.zeros(maxiter) #Vector Alpha
    beta = np.zeros(maxiter) #Beta Alpha  at the index 0 will be  Beta_2 
    res=[] #residual array
    #Lanczos Iteraions
    x1=b.copy() # spero che sia solo per inizializzarlo con le giuste dimensioni 
    for j in range(maxiter):
        
        #(B_jq_j+1)w = Aq_j - B_j-1 q_j-1 - alpha_jq_j
        w = func(A,Q[:,j])   #w = Aq_j
        if j > 0 : 
            w-= beta[j-1] * Q[:,j-1]  #w - B_j-1 q_j-1
        alpha[j] = np.dot(Q[:,j], w) # alpha_j= qj.T Aqj (wj) 
        #wj
        w-=np.dot( Q[:,j], alpha[j]) #w - alpha_j q_j
        
        beta[j] = np.linalg.norm(w) #  beta_j+1 = ||w_j||_2
        q_next=np.divide(w , beta[j])
        Q[:, j+1] = q_next  #qj+1= w/beta_j
        
        if j > 0 : 
            v=np.eye(j+1,1) * b_norm
            T[j,j] = alpha[j]
            T[j-1,j] = beta[j-1]
            T[j+1,j] = beta[j]
        else:
            v=np.eye(2,1) * b_norm
            T[0,0] = alpha[j]
            T[1,0] = beta[j]
        ###TO DO
        x1= x + alpha[j] * w    
        if isclose(beta[j],0.0):
            exit=exitmsgs[2].format(j=j)
            break    
        #print("y numpy:",np.linalg.lstsq(H,v,rcond=None)[0])
        
        if j > 0 : 
            y = QR_ls(T[:j+1,:j+1],v) 
        else:
            y = QR_ls(T[:2,:1],v)
            
        x = np.dot(Q[:,:j+1],y)
        r = np.subtract(np.reshape(b,(len(b),1)),np.dot(A,x))
        r = np.linalg.norm(r)
        res.append(r)
        if r < tol:
            exit=exitmsgs[1].format(j=j)
            break
    #return Q_j and not Q_j+1
    return Q[:,:-1],T[:j+1,:j+1],x,j+1,res,exit

def lanczos_minres(A, b , x0=None,tol=1e-5, maxiter=None, func=__prodMV):
    #initialize variable
    n = len(b)
    if maxiter== None:
        maxiter = n * 5
    
    exitmsgs = [
        "Exit Minres: A  does not define a symmetric matrix",
        "A solution to Ax = b was found at given tol at iteration {j}",
        "Lucky Breakdown solution of Ax=b with Beta_j=0 was found at iteration {j}"
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
        x = np.zeros(A.shape[0])
        r = b.copy()
    else:
        x = x0
        r = np.subtract(np.reshape(b,(len(b),1)),np.dot(A,x0))
    b_norm=np.linalg.norm(b)
    # beta1=np.linalg.norm(r) # non so se sia corretto nel caso in cui il residuo sua diverso da b
    Q = np.zeros((len(b),maxiter))
    P = np.zeros((len(b),maxiter))
    Q[:,0] = np.divide(r,np.linalg.norm(r)) #this can be considered as Beta_1 np.linalg.norm(b) and b as w_1
    T = np.zeros((maxiter+1,maxiter)) #Triadiagonal Matrix
    alpha = np.zeros(maxiter) #Vector Alpha
    beta = np.zeros(maxiter) #Beta Alpha  at the index 0 will be  Beta_2 
    res=[] #residual array
    #Lanczos Iteraions
    x1=b.copy() # spero che sia solo per inizializzarlo con le giuste dimensioni 
    for j in range(maxiter - 1):
        
        #(B_jq_j+1)w = Aq_j - B_j-1 q_j-1 - alpha_jq_j
        w = func(A,Q[:,j])   #w = Aq_j per me qui ci deve stare un'altra variabile
        if j > 0 :  # altrimenti perdo r iniziale 
            w-= beta[j-1] * Q[:,j-1]  #w - B_j-1 q_j-1
        alpha[j] = np.dot(w,Q[:,j]) # alpha_j= qj.T Aqj (wj) 
        #wj
        w-=np.dot(alpha[j],Q[:,j]) #w - alpha_j q_j
        
        beta[j] = np.linalg.norm(w) #  beta_j+1 = ||w_j||_2
        q_next=np.divide(w , beta[j])
        Q[:, j+1] = q_next  #qj+1= w/beta_j
        
        if j > 0 : 
            # v=np.eye(j+1,1) * b_norm
            T[j,j] = alpha[j]
            T[j-1,j] = beta[j-1]
            T[j+1,j] = beta[j]
        else:
            # v=np.eye(2,1) * b_norm
            T[0,0] = alpha[j]
            T[1,0] = beta[j]
        
        F,R=givens_rotation(T) #Q R
        # R_ = np.reshape(R,(R.shape[0]-1,R.shape[1])).copy()
        # print(f'P shape {P.shape}')
        # print(f'R_ shape {R_.shape}')
        # print(f'Q shape {Q.shape}')
        # print(R[j][j])
        # print(R[j,j])
        # print(R[:,j])
        # print('++++++++++++++++++++++++++++++++++++++++++++++')
        # print(R[:][j])
        # print(R[:,:])
        # input('premi')
        if(j >= 2): 
            P[:,j] = np.divide((Q[:,j] - np.dot(R[j-1,j],P[:,j-1]) - np.dot(R[j-2,j],P[:,j-2])),R[j,j])
        elif(j == 1):
            P[:,j] = np.divide((Q[:,j] - np.dot(R[j-1,j],P[:,j-1])),R[j,j])
        else:
            P[:,j] = np.divide(Q[:,j],R[j,j])

        if isclose(beta[j],0.0):
            exit=exitmsgs[2].format(j=j)
            break    
        #print("y numpy:",np.linalg.lstsq(H,v,rcond=None)[0])
        
        # if j > 0 : 
        #     y = QR_ls(T[:j+1,:j+1],v) 
        # else:
        #     y = QR_ls(T[:2,:1],v)

        # x = np.dot(Q[:,:j+1],y)
        # print(f'F_j: {F[0][j]} P_j: {P[:][j]} v[0][0]: {v[0][0]}')
        # input('premi per andare avanti')
        x += np.dot(np.dot(b_norm,F[0,j]),P[:,j]) 
        r = np.subtract(np.reshape(b,(len(b),1)),np.dot(A,x))
        r = np.linalg.norm(r)
        print(f'residual: {r}')
        res.append(r)
        if r < tol:
            exit=exitmsgs[1].format(j=j)
            break
    #return Q_j and not Q_j+1
    return Q[:,:-1],T[:j+1,:j+1],x,j+1,res,exit
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


def SymOrtho(a, b):
    if b == 0:
        s = 0
        r = abs(a)
        if a == 0:
            c = 1 
        else:
            c = np.sign(a)
    elif a == 0:
        c = 0
        s = np.sign(b)
        r = abs(b)
    elif abs(b) > abs(a):
        t = a/b
        s = np.sign(b)/hypot(1,t)
        c = s*t 
        r = b/s
    elif abs(a) > abs(b):
        t = b/a
        c = np.sign(a)/hypot(1,t)
        s = c*t
        r = a/c
        
    return c, s, r

def lanczos(A,v1,v0,beta1,func=__prodMV):
    pk= func(A,v1)   #p = Av_j 
    
    alpha = np.dot(v1, pk) # alpha_j = vj.T Avj (Avj=pk) 
    
    pk-= alpha * v1 #w - alpha_j q_j
    
    v_next = pk - beta1 * v0 
    
    beta2 = np.linalg.norm(v_next) #  beta_j+1 = ||w_j||_2
    
    if not isclose(beta2,0.0):
        v_next= np.divide(v_next , beta2) #v_next normalization
    
    return alpha,beta2,v_next
    
    
def minresSlide(A,b,maxit):
    beta1=np.linalg.norm(b)
    v0=0
    v1=np.divide(b,beta1)
    c0=-1
    s0=0
    d1=d0=x=np.zeros(len(b))
    k=1
    delta1=eps=0
    phi0=beta1
    res=[]
    while k < maxit:
       alphak,beta2,v_next=lanczos(A,v1,v0,beta1)
       delta2= c0*delta1 + s0 * alphak
       gamma = s0 * delta1 - c0 * alphak
      
       delta1_next = -c0 * beta2
       ck,sk,gamma=SymOrtho(gamma,beta2)
       tk = ck*phi0
       phik=sk*phi0
       
       if gamma!=0:
            dk=np.divide((v1-delta2*d1-eps*d0),gamma)
       x = x+ tk * dk
       eps = s0 * beta2
       phi0=phik
       delta1=delta1_next
       c0=ck
       s0=sk
       beta1=beta2
       v0=v1
       v1=v_next
       d0=d1
       d1=dk
       k=k+1
       
       #Computation of the residual
       r = np.subtract(np.reshape(b,(len(b),1)),np.dot(A,x))
       r = np.linalg.norm(r)
       print(r)
       res.append(r)
    return k,res