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

def read_faces():
   M = 192*168
   N = 38 
   A = numpy.zeros((M,N))
   for i in range(N):
      try:
         im = Image.open("faces/face"+"%02d"%(i+1)+".pgm")
         im = im.convert('L')
         mat = numpy.array(im.getdata())
         A[:,i] = mat
      except:
         pass
   return A

def read_one_face():
   M = 192*168
   N = 64 
   A = numpy.zeros((M,N))
   for i in range(N):
      try:
         im = Image.open("yaleB01/face"+"%02d"%(i+1)+".pgm")
         im = im.convert('L')
         mat = numpy.array(im.getdata())
         A[:,i] = mat
      except:
         print "Could not find face number",i+1
   return A

"""
Display RPCA results for the i-th frame/image.
Figure 1: original frame
Figure 2: low-rank component
Figure 3: sparse component ("outliers")
"""
from pylab import *
def showframe(A,X,Y,i,shape):
    matshow(A[:,i].reshape(shape),cmap=cm.gray)
    matshow(X[:,i].reshape(shape),cmap=cm.gray)
    matshow(Y[:,i].reshape(shape),cmap=cm.gray)
    show()

