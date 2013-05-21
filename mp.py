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



def omp(A,s,epsilon, L=None):
   """
   Orthogonal Matching Pursuit
   Solve the problem:
      min ||x||_0 s.t. ||Ax - s|| <= epsilon

   Select the best new columns, update all the coefficients (not just the new one)
   """
   #TODO vectorized S

   (M,N) = A.shape
   P = s.shape[1] # Number of signals. Only works for P=1, at the moment!
   x = zeros((N,P))
   
   xs = [x]
   r = s
   max_it = 1000
   i = 0
   
   if L is None:
      L = N

   columns = []
   while i <= max_it and linalg.norm(r) >= epsilon and len(columns) <= L:
      k_hat = argmax([abs(dot(A[:,k],r)/linalg.norm(A[:,k])) for k in range(N)])
      columns.append(k_hat)

      # Select columns corresponding to nonzero coefficients
      A_w = A.take(columns,axis=1)
      # Compute coefficient for selected columns
      x_sparse = dot(dot(linalg.inv(dot(A_w.T,A_w)),A_w.T),s) 
      # Construct new solution
      x_star = zeros((N,1))
      for i,idx in enumerate(columns):
         x_star[idx] = x_sparse[i]
      xs.append(x_star)
      
      i += 1
      r = s - dot(A,xs[i])
      #print i,k_hat, linalg.norm(r)

   return xs[i]
