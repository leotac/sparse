"""
Algorithms for dictionary learning in sparse representation.
"""
from numpy import *
import mp
import sparse
import scipy.sparse.linalg

def ksvd(B,S,max_iter=10):
   M,N = B.shape
   P = S.shape[1]
   #print P 
   A = B.copy()
   X = zeros((N,P))

#   max_iter = 10
   i = 0
   while i <= max_iter:
      
      for p in range(P):
         X[:,p] = mp.omp(A,S[:,p].reshape(M,1),0.01).flatten()
      print A,X
      print "-----------------"
      print concatenate((S,dot(A,X)),axis=1) 
      for l in range(N):
         E = S - dot(A,X) + dot(A[:,l].reshape(M,1),X[l,:].reshape(1,P))
         print "S,AX", concatenate((S,dot(A,X)),axis=1) 
         print dot(A[:,l].reshape(M,1),X[l,:].reshape(1,P)) 
         #print "hhhhhhhhhh"
         u,s,v = scipy.sparse.linalg.svds(E, k = 1)
         print "e",E
         print "USV", u,s,v
         A[:,l] = u.flatten()
         X[l,:] = s*v.flatten()


      i += 1
      print A,X

   return A, X
