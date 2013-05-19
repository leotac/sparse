from numpy import *
import scipy as sp
import scipy.linalg
import time 

def time_it(func):
   def wrapper(*arg):
      t1 = time.time()
      res = func(*arg)
      t2 = time.time()
      #print '%s took %0.3f ms' % (func.func_name, (t2-t1)*1000.0)
      return (t2-t1)*1000.0
   return wrapper

def get_rand(M,N):
   #M,N = 5,10
   A = random.random((M,N))
   s = random.random((N,1))
   p = 0.5
   return A,s,p

# Find x s.t. (A'A + p*I)x = s

def run_test(M,N,x):
   funs = [direct_solve, inverse_solve, lemma_solve]
   t = []
   for fun in funs:
      t.append(run_fun(M,N,fun,x,1))
   return t
      

def run_fun(M,N,fun,x,seed=1):
   random.seed(seed)
   times = []
   for i in range(x):
      A,s,p = get_rand(M,N)
      times.append(fun(A,s,p)) 
   return mean(times)


@time_it
def direct_solve(A,s,p):
   # Solve directly
   (M,N) = A.shape
   x = linalg.solve(dot(A.T,A) + p*eye(N),s)
   return x

@time_it
def inverse_solve(A,s,p):
   (M,N) = A.shape
   x = dot(linalg.inv(dot(A.T,A) + p*eye(N)),s)
   return x

@time_it
def LU_solve(A,s,p):
   (M,N) = A.shape
   return 0

@time_it
def lemma_solve(A,s,p):
   # Solve with matrix inversion lemma
   (M,N) = A.shape
   C = dot(A,A.T) + p*eye(M)
   P,L,U = sp.linalg.lu(C)
   y = linalg.solve(L,dot(A,s))
   z = linalg.solve(U,y)
   x = (1/p)*s - (1/p)*dot(A.T,z)
   return x


if __name__ == '__main__':
    times = [run_test(20,k*20,10) for k in range(1,21)]
    import pylab
    pylab.plot(times)
    pylab.show()

