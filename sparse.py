from numpy import *

"""
Utility function for sparse representation
"""

def zero_norm(x, epsilon = 0.00001):
   """
   Compute 0-norm with precision epsilon
   """
   return sum(1 for i in x if abs(i) >= epsilon)

def zero_idx(x, epsilon = 0.00001):
   """
   Get indices of zero elements 
   """
   return [i for i in range(len(x)) if abs(x[i]) < epsilon]

def spark(A):
   """
   Get spark of matrix A, defined as min(N,rank(A)+1)
   """
   return min(A.shape[1],linalg.matrix_rank(A) + 1)

def mu(A):
   """
   Get mutual coherence
      mu(A) = max{k,l} (a_k'*a_l) / (||a_k||_2*||a_l||_2)
   """
   return max([dot(A[:,l],A[:,k])/(norm(A[:,l])*norm(A[:,k])) for k in range(N) for l in range(N) if l>k]) 


def nonzero_idx(x, epsilon = 0.00001):
   """
   Get indices of nonzero elements 
   """
   return [i for i in range(len(x)) if abs(x[i]) >= epsilon]


def shrinkage(z,gamma):
   """
   Shrinkage
   """
   return sign(z) * maximum(zeros(z.shape), abs(z) - gamma)


def gen_rand_inst(M, N, noise=False):
   """
   Generate a MxN dictionary, a random sparse vector x 
   and the reconstructed signal s.
   If noise=True, noise is added to s (thus A*x != s)
   """
   A = random.random((M,N))
   x = random.random((N,1))
   for (i,val) in enumerate(x):
      if random.random() >= 0.2:
         x[i] = 0
   nu = 0
   if noise:
      nu = 0.03*random.random((M,1))
   s = dot(A,x) + nu
   return A,x,s

def gen_P_inst(M, N, P, noise=False):
   """
   Generate a MxN dictionary and P random signals s.
   """
   A = random.random((M,N))
   S = random.random((M,P))
   return A,S


class DCT(object):

   def __init__(self):
      pass
       
   def transKernel(self,N):
      A = zeros((N,N))
      for k in xrange(0,N):
         for n in xrange(0,N):
            if n == 0:
               A[n,k] = sqrt(1/float(N)) 
            else:
               A[n,k] = sqrt(2/float(N))*cos(pi*(2*k+1)*n/float(2*N))
      return A
   
   def transOverKernel(self,M,N):
      A = zeros((M,N))
      for k in xrange(0,N):
         for n in xrange(0,M):
            if n == 0:
               A[n,k] = sqrt(1/float(N)) 
            else:
               A[n,k] = sqrt(2/float(N))*cos(pi*(2*k+1)*n/float(2*N))
      return A
      
   def transform(self,s):
       #(M,N) = m.shape 
       tk = self.transKernel(len(s))
       t1 = dot(tk,s)
       return t1
