from numpy import *
import scipy
import solve

def ADMM_BPDN(A,s,l):
   """
   Alternating Direction Method of Multipliers for Basis Pursuit Denoising
   Solve the problem:
      argmin 1/2 ||Ax-s||^2 + lambda*||x||_1
   """
   (M,N) = A.shape
   x,y,u = zeros((N,1)), zeros((N,1)), zeros((N,1))
   max_it = 100
   rho = l 
   k = 0
   xks,yks,uks = [x],[y],[u]

   # Cached computations
   C = dot(A,A.T) + rho*eye(M)
   PL,U = scipy.linalg.lu(C, permute_l=True)
   As = dot(A.T,s)

   while k <= max_it:
      # Update x
      rhs = As + rho*(yks[k] - uks[k])
      x_star = solve.lemma_solve(A,rhs,rho,PL,U)
      xks.append(x_star)
      
      # Update y
      z = xks[k+1] + uks[k]
      y_star = sign(z) * maximum(zeros(z.shape),abs(z) - l/rho)
      yks.append(y_star)
   
      # Update u
      u_star = uks[k] + xks[k+1] - yks[k+1]
      uks.append(u_star)
      
      print k, 1/2*linalg.norm(dot(A,x_star)-s) + l*linalg.norm(x_star,1)
      k += 1
      
   return xks[k-1]


def gen_rand_inst(M, N, noise=False):
   """
   Generate a MxN dictionary, a random sparse vector x 
   and the reconstructed signal s.
   If noise=True, noise is added to s (thus A*x != s)
   """
   A = random.random((M,N))
   x = random.random((N,1))
   for (i,val) in enumerate(x):
      if random.random() >= 0.5:
         x[i] = 0
   nu = 0
   if noise:
      nu = 0.03*random.random((M,1))
   s = dot(A,x) + nu
   return A,x,s

