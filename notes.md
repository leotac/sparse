RPCA
====

- ALM instead of ADMM [see paper] - [?]
- Matrix completion? Adapt RPCA-ADMM - [ ]

SVD in Python
-------------
- Sparse version with estimated number of singular values - [X] no apparent improvement both in SVD time and overall efficiency
- Better optimized BLAS - Not really doable on a VM
- Distributed gensim - Not really doable
- SVDLIBC  [seems even slower than scipy.sparse.svds]
- https://github.com/jakevdp/pypropack [not tried]
- http://fa.bianp.net/blog/2012/singular-value-decomposition-in-scipy/
- http://jakevdp.github.io/blog/2012/12/19/sparse-svds-in-python/


Dictionary Learning
=================
- How to test KSVD
- Other methods

Basis Pursuit
============
- Constrained version: test, dbg
- Apply BP to classification - [X]. Slow, but it seems to work.  
   Use 'labeled' pictures as dictionary atoms.  
   Use picture to classify as 's'.  
   Recover sparse x_str [takes a while..].  
   Get label of argmax over x - or some more sofisticated measure.  


(O)MP
=====
- MMV version
- Optimize
