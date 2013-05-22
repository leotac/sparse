"""
Algorithms for dictionary learning in sparse representation.
"""
from numpy import *
import mp
import sparse
import scipy.sparse.linalg

def ksvd(AA,S,max_iter=10):
   M,N = AA.shape
   P = S.shape[1]
   #print P 
   A = AA.copy()
   X = zeros((N,P))

   #max_iter = 10
   i = 0
   while i <= max_iter:
      
      # SMV
      for p in range(P):
         X[:,p] = mp.omp(A,S[:,p].reshape(M,1),0.1).flatten()
      #print A,X
      print "-----------------"
      print i,linalg.norm(S - dot(A,X))
      #print X, "--" 
      print sum([sparse.zero_norm(X[:,p],0.01) for p in range(P)]) 
      print "--"

      for l in range(N):
         #E = S - dot(A,X) + dot(A[:,l].reshape(M,1),X[l,:].reshape(1,P))

         nonzeros = sparse.nonzero_idx(X[l,:]) # xs witth nonzero in row 'l'
         if len(nonzeros) == 0:
            continue
         E_w = S[:,nonzeros] - dot(A,X[:,nonzeros]) + dot(A[:,l].reshape(M,1),X[l,nonzeros].reshape(1,len(nonzeros)))

         #print "S,AX", concatenate((S,dot(A,X)),axis=1) 
         #print dot(A[:,l].reshape(M,1),X[l,:].reshape(1,P)) 
         
         #u,s,v = scipy.sparse.linalg.svds(E, k = 1)
         u,s,v = scipy.sparse.linalg.svds(E_w, k = 1)
         #print "e",E
         #print "USV", u,s,v
         A[:,l] = u.flatten()
         X[l,nonzeros] = s*v.flatten()


      print "--"
      i += 1

   return A, X
