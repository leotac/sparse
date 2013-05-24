"""
Algorithms for dictionary learning in sparse representation.
"""
from numpy import *
import mp
import sparse
import scipy.sparse.linalg
import sklearn.linear_model

def ksvd(AA,S,max_iter=10):
"""
Apply KSVD from the starting dictionary AA, given a set of signals S.
"""   
   M,N = AA.shape
   P = S.shape[1]
   A = AA.copy()
   X = zeros((N,P))

   #max_iter = 10
   i = 0
   while i < max_iter:
      # SMV, and slow.
      #for p in range(P):
      #   X[:,p] = mp.omp(A,S[:,p].reshape(M,1),0.1).flatten()
     
      # MMV
      # Default: no residual tolerance, max nonzeros 10%
      # Assume dictionary A is normalized! 
      X = sklearn.linear_model.orthogonal_mp(A,S)
      
      print i,linalg.norm(S - dot(A,X)), sum([sparse.zero_norm(X[:,p],0.01) for p in range(P)]) 

      for l in range(N):
         nonzeros = sparse.nonzero_idx(X[l,:]) # xs with nonzero in row 'l'
         if len(nonzeros) == 0:
            continue
         E_w = S[:,nonzeros] - dot(A,X[:,nonzeros]) + dot(A[:,l].reshape(M,1),X[l,nonzeros].reshape(1,len(nonzeros)))

         try:
            u,s,v = scipy.sparse.linalg.svds(E_w, k = 1)
         except:
            continue
         A[:,l] = u.flatten()
         X[l,nonzeros] = s*v.flatten()

      i += 1

   return A, X
