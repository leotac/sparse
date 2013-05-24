"""
Application of Alternating Direction Method of Multipliers 
to problems in Sparse Representation

"""
from numpy import *
import scipy
import solve
import sparse

def ADMM_BPDN(A,s,l, max_it=1000):
   """
   Alternating Direction Method of Multipliers for Basis Pursuit Denoising
   Solve the problem:
      min 1/2 ||Ax-s||^2 + lambda*||x||_1
   transformed into:
      min 1/2 ||Ax-s||^2 + lambda*||y||_1 s.t. x = y
   """
   (M,N) = A.shape
   x,y,u = zeros((N,1)), zeros((N,1)), zeros((N,1))
   rho = l 
   k = 0
   xks,yks,uks = [x],[y],[u]
   
   assert(s.shape == (M,1))
   
   # Cached factorizations when M < N
   if M < N:
      C = dot(A,A.T) + rho*eye(M)
      PL,U = scipy.linalg.lu(C, permute_l=True)
   
   As = dot(A.T,s)

   prim_resid = 100
   dual_resid = 100
   prim_epsi = 0
   dual_epsi = 0
   epsi_rel = 0.001
   epsi_abs = 0.0001
   
   while k < max_it and (prim_resid >= prim_epsi or dual_resid >= dual_epsi):
      # Update x
      rhs = As + rho*(yks[k] - uks[k])
      if M < N:
         x_star = solve.lemma_solve(A,rhs,rho,PL,U)
      else:
         x_star = solve.direct_solve(A,rhs,rho)
      xks.append(x_star)
      
      # Update y
      z = xks[k+1] + uks[k]
      y_star = sparse.shrinkage(z,  l/rho)
      yks.append(y_star)
   
      # Update u
      u_star = uks[k] + xks[k+1] - yks[k+1]
      uks.append(u_star)
      
      prim_resid = linalg.norm(xks[k+1] - yks[k+1])
      dual_resid = linalg.norm(rho*(yks[k] - yks[k+1]))
      #TODO change rho to balance primal/dual residual
      print k, 1/2*linalg.norm(dot(A,x_star)-s) + l*linalg.norm(x_star,1)
 
      k += 1
      prim_epsi = sqrt(N)*epsi_abs + epsi_rel*max(linalg.norm(xks[k]),linalg.norm(yks[k]))
      dual_epsi = sqrt(N)*epsi_abs + epsi_rel*linalg.norm(uks[k])
      
      print prim_resid, dual_resid 
      print prim_epsi, dual_epsi
 
   return xks[k]


def ADMM_ConstrBP(A,s,epsilon):
   """
   Alternating Direction Method of Multipliers for Constrained Basis Pursuit
   Solve the problem:
      min ||x||_1 s.t. ||Ax-s||_2 <= epsilon
   i.e., find the minimum-norm vector s.t. the reconstruction error is under epsilon.
   The problem is transformed into:
      min ||y||_1 + 1/2||z-s||_2 s.t x = y, Ax = z
   """
   (M,N) = A.shape
   x,y,z,u,v = zeros((N,1)), zeros((N,1)), zeros((M,1)), zeros((N,1)), zeros((M,1))

   max_it = 200
   
   gamma = epsilon
   sigma = epsilon

   k = 0
   xks,yks,zks,uks,vks = [x],[y],[z],[u],[v]

   prim_resid = 100
   dual_resid = 100
   prim_epsi = 0
   dual_epsi = 0
   epsi_rel = 0.001
   epsi_abs = 0.0001
   
   while k < max_it: #TODO stopping criterion
      # Update x
      rhs = gamma*(yks[k] - uks[k]) + sigma*dot(A.T,(zks[k]-vks[k]))
      x_star = solve.lemma_solve(A,rhs,gamma,PL,U)
      xks.append(x_star)
      
      # Update y
      t = xks[k+1] + uks[k]
      y_star = sparse.shrinkage(t, 1/gamma)
      yks.append(y_star)
  
      # Update z
      b = dot(A,xks[k+1]) + vks[k]
      if linalg.norm(b-s) > epsilon:
         z_star = s + epsilon*(b - s)/linalg.norm(b-s) 
      else:
         z_star = b
      zks.append(z_star)

      # Update u
      u_star = uks[k] + xks[k+1] - yks[k+1]
      uks.append(u_star)
 
      # Update v
      v_star = vks[k] + dot(A,xks[k+1]) - zks[k+1]
      vks.append(v_star)
     
      prim_resid = linalg.norm(xks[k+1] - yks[k+1])
      dual_resid = linalg.norm(rho*(yks[k] - yks[k+1]))

      print k, linalg.norm(x_star,1)
      print linalg.norm(dot(A,x_star)-s)
 
      k += 1
      prim_epsi = sqrt(N)*epsi_abs + epsi_rel*max(linalg.norm(xks[k]),linalg.norm(yks[k]))
      dual_epsi = sqrt(N)*epsi_abs + epsi_rel*linalg.norm(uks[k])
      
      print prim_resid, dual_resid 
      print prim_epsi, dual_epsi
 
   return xks[k]



def ADMM_RPCA(D, max_it=10):
   """
   Alternating Direction Method of Multipliers for 
   Robust Principal Component Analysis (a.k.a. Principal Component Pursuit)
      min ||X||_* + lambda*||Y||_1 s.t. X + Y = D

   Original source: Candes, E. J., Li, X., Ma, Y., & Wright, J. (2009). Robust principal component analysis?. arXiv preprint arXiv:0912.3599.
   An implementation: http://www.cds.caltech.edu/~ipapusha/code/pcp_admm.m
   """
   (M,N) = D.shape
   
   l = 1 / sqrt(max(M,N))
   print "Lambda", l
   rho = 10*float(M)*N/(4*sparse.one_norm(D))
   print "Rho:", rho
   k = 0

   X,Y,Z = zeros((M,N)), zeros((M,N)), zeros((M,N))
   
   epsilon = 0.0000001 * linalg.norm(D)
   residual = epsilon * 100
   print "Primal tolerance", epsilon
   print "Max iterations", max_it 

   while k < max_it and residual >= epsilon:
      # Update X
      print "SVD it"
      U,sigma,Vh = linalg.svd(D - Y - Z, full_matrices=False)
      print "Get X"
      X = dot(dot(U, diag(sparse.shrinkage(sigma,1/rho))),Vh)
     
      # Update Y
      # Element-wise shrinkage
      print "Shrink it"
      newY = sparse.shrinkage(D - X - Z,  l/rho)
   
      # Update Z 
      Z = Z + X + newY - D
     
      print "Norm it"
      residual = linalg.norm(D - X - Y)
      dual_residual = rho*linalg.norm(newY - Y)
      Y = newY
      #print k, 1/2*linalg.norm(dot(A,x_star)-s) + l*linalg.norm(x_star,1)
      #TODO vary rho
      #TODO dual_tolerance
      
      k += 1
      
      print k, residual, dual_residual, sum(Y!=0) 
 
   return X, Y



#from sparsesvd import sparsesvd
def ADMM_sparse_RPCA(D, max_it=10):
   """
   Alternating Direction Method of Multipliers for 
   Robust Principal Component Analysis (a.k.a. Principal Component Pursuit)
      min ||X||_* + lambda*||Y||_1 s.t. X + Y = D
   
   Sparse SVD: compute only the first 'sv' singular values, where sv is
   an estimate of the number of singular values that are greater than
   the value 1/rho, used in the shrinkage operation.

   A different approach with sparse SVD: Lin, Z., Chen, M., & Ma, Y. (2010). The augmented lagrange multiplier method for exact recovery of corrupted low-rank matrices. arXiv preprint arXiv:1009.5055.
   """
   (M,N) = D.shape
   X = zeros((M,N))
   Y = zeros((M,N))
   Z = zeros((M,N))

   l = 1 / sqrt(max(M,N))
   print l
   rho = float(M)*N/(4*sparse.one_norm(D))
   print rho
   k = 0
   epsilon = 0.0000001 * linalg.norm(D)
   residual = epsilon * 100
   print epsilon
   sv = 20
   while k < max_it and residual >= epsilon:
      # Update X
      print "CSC it"
      TMP = scipy.sparse.csc_matrix(D-Y-Z)
      print "SVD it"
      U,sigma,Vh = scipy.sparse.linalg.svds(TMP, k = sv)
      #Ut,sigma,Vh = sparsesvd(TMP, k = sv)
      print sv, sum(sigma >= 1/rho)
      if sum(sigma >= 1/rho) < sv:
         sv += 1
      else:
         sv = int(min(sv + (k+1)*0.05*N,N))

      print "Get X"
      X = dot(dot(U, diag(sparse.shrinkage(sigma,1/rho))),Vh)

      # Update Y
      # Element-wise shrinkage
      print "Shrink it"
      Y = sparse.shrinkage(D - X - Z,  l/rho)
   
      # Update Z 
      Z = Z + X + Y - D
     
      print "Norm it"
      residual = linalg.norm(D-X-Y)

      #print k, 1/2*linalg.norm(dot(A,x_star)-s) + l*linalg.norm(x_star,1)
 
      k += 1
      
      print k, residual, sum(Y!=0) 
 
   return X, Y

