import numpy as np

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

def myQR(A,slower=False):
    m,n=A.shape
    Q = np.eye(m)
    R= A.copy()
    for j in (range(n)):
        u,s=householder_vector(R[j:,j])
        u.resize((len(u),1))
        H = np.eye(len(u)) - np.dot((2*u),u.T);
        if not slower:
            R[j:,j:]= R[j:,j:] - np.dot(2*u,np.dot(u.T,R[j:,j:]))
        else:
            R[j:,j:]= H  @ R[j:,j:]
        Q[:,j:]= Q[:,j:]@H    
    return Q,R

def QR_ls(A: np.ndarray, b: np.ndarray):
    Q,R=myQR(A)
    #Rx=Q.Tb
    m,n=R.shape
    c=np.dot(Q[:,:n].T,b)
    x=back_substitution(R[:n,:n],c)
    print("x",x)
    return x

def __prodMV(A,v): 
    return np.matmul(A,v)

def lanczos_minres(A, b , x0=None,tol=1e-5, maxiter=None, func=__prodMV):
    #initialize variable
    n = len(b)
    j=0

    if maxiter== None:
        maxiter = n * 5
    
    exitmsgs=["Exit Minres: A  does not define a symmetric matrix","A solution to Ax = b was found, given tol at iteration {j}","A solution to Ax = b was found at iteration {j}"]
    
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
        w = b.copy()
    else:
        w = np.subtract(np.reshape(b,(len(b),1)),np.dot(A,x0))
    
    beta1=np.linalg.norm(w)
    Q = np.zeros((len(b),maxiter+1))
    Q[:,0] = w/beta1 #this can be considered as Beta_1 np.linalg.norm(b) and b as w_1
    
    alpha = np.zeros(maxiter)
    beta = np.zeros(maxiter)
    res=[]
    for j in range(maxiter):
        #(B_jq_j+1)w = Aq_j - B_j-1 q_j-1 - alpha_jq_j
        w = func(A,Q[:,j])   #w = Aq_j
        if j > 0 : 
            w-= beta[j-1] * Q[:,j-1]  #w - B_j-1 q_j-1
        alpha[j] = np.dot(Q[:,j], w) # alpha_j= qj.T Aqj (wj) 
       #wj
        w-=np.dot( Q[:,j], alpha[j]) #w - alpha_j q_j
        beta[j]= np.linalg.norm(w) #  beta_j= ||w_j||_2
        q_next=np.divide(w , beta[j])
        Q[:, j+1] = q_next  #qj+1= w/beta_j
        
        if j > 0 : 
            v=np.eye(j+1,1) * np.linalg.norm(b)
            rbeta = beta[0:-1] #without the last row of Beta_j 
            H = np.diag(alpha)+ np.diag(rbeta, +1) + np.diag(rbeta,-1)
            H = H[:j+1,:j+1]
        else:
            v=np.eye(2,1) * np.linalg.norm(b)
            H = np.zeros((2,1))
            H[0] = alpha[j]
            H[1] = beta[j]
        #print("y numpy:",np.linalg.lstsq(H,v,rcond=None)[0])
        y = QR_ls(H,v)
        x = np.dot(Q[:,:j+1],y)
        r = np.subtract(np.dot(A,x),np.reshape(b,(len(b),1)))
        r = np.linalg.norm(r)
        res.append(r)
        if r < tol:
            if(beta[j]==0):
                exit=exitmsgs[2].format(j=j)
            else:
                exit=exitmsgs[1].format(j=j)
            break
    #return Q_j and not Q_j+1
    return Q[:,:-1],H,x,j,res,exit

'''def MINRES2(A,b:np.ndarray,N, func=__prodMV):
    exit=N
    k=N
    print(N)
    j:np.ndarray
    Q:np.ndarray = np.zeros((len(b),k+1))
    Q[:,0]=np.divide(b,np.linalg.norm(b))
    H:np.ndarray = np.zeros((k+1,k))
    for j in range(k):
        w = A @ Q[:,j] # A * r
        for i in range(j): 
            H[i,j] = func(Q[:,i].T,w)
            w = w - np.dot(Q[:,i] , H[i,j])
        H[j+1,j] = np.linalg.norm(w)
        Q[:,j+1]= np.divide(w,H[j+1,j])
        print('Iter:',j, 'residual:',np.linalg.norm(Q[:,j+1]))
        if np.linalg.norm(Q[:,j+1])<1e-10:    
            exit=j
            print('Iter:',j, 'residual:',np.linalg.norm(Q[:,j+1]))
            break 
        # stack=np.divide(Ar.copy() , beta[m+1]) or np.linalg.norm(alpha*Ar)<= 0
        # stack=np.reshape(stack,(len(stack),1))
        # Q = np.hstack((Q,stack))
    print('Iter:',0, 'residual:',np.linalg.norm(Q[:,0]))
    input('premi per continuare')
    # rbeta = beta[1:-1]
    # H = np.diag(alpha)+ np.diag(rbeta, +1) + np.diag(rbeta,-1)
    return Q,exit
'''

#use lib to find eigenvalues of matix H
def eigenvalues(H):
    return np.linalg.eig(H)[0]

#print eigenvalut min and max given a egignevalue  list
def print_first_last(a):
    print("Min  ",'%1.9g\t' % a.min())
    print("Max", '%1.9g' % a.max())
    
#print the firt N eigenvalue
def print_first_N(a,N):
    try: acopy = a.copy()
    except: acopy = a[:]
    acopy.sort()
    max = min(N,len(a))
    for i in range(max):
        print('%1.5g\t' % acopy[i])
    print('')
