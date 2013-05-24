from numpy import *
import scipy as sp
import scipy.linalg
import time 

def time_it(func,*arg):
   t1 = time.time()
   res = func(*arg)
   t2 = time.time()
   return (t2-t1)*1000.0

def get_rand(M,N):
   #M,N = 5,10
   A = random.random((M,N))
   s = random.random((N,1))
   p = 0.5
   return A,s,p


def run_test(M,N,x,funs):
   t = []
   for fun in funs:
      t.append(run_fun(M,N,fun,x,1))
   return t
      

def run_fun(M,N,fun,x,seed=1):
   random.seed(seed)
   times = []
   for i in range(x):
      A,s,p = get_rand(M,N)
      times.append(time_it(fun,A,s,p)) 
   return mean(times)

##################################
#
# Find x s.t. (A'A + p*I)x = rhs
#
##################################
def direct_solve(A,rhs,p):
   # Solve directly
   (M,N) = A.shape
   return linalg.solve(dot(A.T,A) + p*eye(N),rhs)

def lstsq_solve(A,rhs,p):
   # Solve directly
   (M,N) = A.shape
   return linalg.lstsq(dot(A.T,A) + p*eye(N),rhs)[0]

def inverse_solve(A,rhs,p):
   (M,N) = A.shape
   return dot(linalg.inv(dot(A.T,A) + p*eye(N)),rhs)

def LU_solve(A,rhs,p):
   # Solve with straight LU
   (M,N) = A.shape
   C = dot(A.T,A) + p*eye(N)
   P,L,U = sp.linalg.lu(C)
   y = linalg.solve(dot(P,L),rhs)
   x = linalg.solve(U,y)
   return x

def lemma_solve(A,rhs,p, PL=None, U=None):
   # Solve with matrix inversion lemma
   # If given, omit L,U decomposition
   (M,N) = A.shape
   if PL == None or U == None:
      PL,U = sp.linalg.lu(dot(A,A.T) + p*eye(M),permute_l=True)
   y = linalg.solve(PL,dot(A,rhs))
   z = linalg.solve(U,y)
   return (1/p)*rhs - (1/p)*dot(A.T,z)

if __name__ == '__main__':
    funs = [direct_solve, inverse_solve, LU_solve, lemma_solve]
    times = [run_test(20,k*20,10,funs) for k in range(1,21)]
    import pylab
    print times,[f.__name__ for f in funs]
    lines = pylab.plot(times)
    pylab.legend(lines,[f.__name__ for f in funs])
    pylab.ylabel("avg time (ms)")
    pylab.xlabel("N/M ratio")
    pylab.show()

