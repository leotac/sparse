Sparse
======

Sparse Representation





Video
-----
import video,admm,solve  
A = video.read_frames()  
X_star,Y_star = admm.ADMM_RPCA(A, 5)  
solve.time_it(admm.ADMM_RPCA,A[:,:80], 5)  
video.showframe(A,X_star,Y_star,60,(360,480))  
