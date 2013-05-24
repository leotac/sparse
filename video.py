"""
Read frames saved as png images
"""
from PIL import Image
import numpy 
def read_frames():
   M = 360*480
   N = 100
   A = numpy.zeros((M,N))
   for i in range(N):
      im = Image.open("frames/frame"+str(i)+".png")
      im = im.convert('L')
      mat = numpy.array(im.getdata())
      A[:,i] = mat
   return A

"""
Display RPCA results for the i-th frame.
Figure 1: original frame
Figure 2: low-rank component
Figure 3: sparse component ("outliers")
"""
import pylab
def showframe(A,X,Y,i,shape):
    pylab.matshow(A[:,i].reshape(shape),cmap=cm.gray)
    pylab.matshow(X[:,i].reshape(shape),cmap=cm.gray)
    pylab,matshow(Y[:,i].reshape(shape),cmap=cm.gray)
    pylab.show()

