from numpy import *
import sparse

def mp(B,s,epsilon):
   """
   Matching Pursuit
   Solve the problem:
      min ||x||_0 s.t. ||Ax - s|| <= epsilon
   Select the best column, update its coefficient
   """
   A = B.copy()
   (M,N) = A.shape
   x = zeros((N,1))
   
   xs = [x]
   r = s
   max_it = 10000
   i = 0

   #for k in range(N):
   #   A[:,k] /= linalg.norm(A[:,k])

   while i <= max_it and linalg.norm(r) >= epsilon:
      k_hat = argmax([abs(dot(A[:,k],r)/linalg.norm(A[:,k])) for k in range(N)])
      x_hat = dot(A[:,k_hat],r)/dot(A[:,k_hat],A[:,k_hat])
 
      xs.append(xs[i])
      xs[i+1][k_hat] += x_hat
      
      i += 1
      
      r = s - dot(A,xs[i])
      print i,k_hat,x_hat, linalg.norm(r)

   return xs[i]



def omp(AA,s,epsilon=None, L=None):
   """
   Orthogonal Matching Pursuit
   Solve the problem:
      min ||x||_0 s.t. ||Ax - s|| <= epsilon

   Select the best new columns, update all the coefficients (not just the new one)
   """
   #TODO vectorized S
   # Very inefficient 

   A = AA.copy()
   (M,N) = A.shape
   # Number of signals. Only works for P=1, at the moment!
   P = s.shape[1]
   x = zeros((N,P))
   
   xs = [x]
   r = s
   max_it = 1000
   i = 0
  
   if epsilon is None:
      epsilon = 0 
      L = N/10

   if L is None:
      L = N
   
   # Normalization
   #for k in range(N):
   #   A[:,k] /= linalg.norm(A[:,k])

   columns = []
   while i <= max_it and linalg.norm(r) >= epsilon and len(columns) < L:
      k_hat = argmax([abs(dot(A[:,k],r)/linalg.norm(A[:,k])) for k in range(N)])
      columns.append(k_hat)
      # Select columns corresponding to nonzero coefficients
      x_sparse, _, _, _ = linalg.lstsq(A[:,columns],s) 
      # Construct new solution
      x_star = zeros((N,1))
      for i,idx in enumerate(columns):
         x_star[idx] = x_sparse[i]
      xs.append(x_star)
      i += 1
      r = s - dot(A[:,columns],x_sparse)
      #print i,k_hat, linalg.norm(r)

   return x_star
