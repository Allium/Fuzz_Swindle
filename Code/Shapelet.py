#!/usr/bin/env/python
##============================================================

import numpy, Image
from sys import argv
#from numpy.polynomial.laguerre import lagval
from scipy.special import eval_genlaguerre

##============================================================

def main(nmax):
	
	arraytoimage(basis_arrays(nmax))
	
	return

##============================================================

"""Compute the radial Gauss-Laguerre functions on a square grid.
Returns a stack of 2D arrays.

Attention: basis scale
"""

def basis_arrays(nmax=0):

##------------------------------------------------------------
	## Important HC-numbers
	
	## Image dimensions in pixels
	w = 25
	h = 25
	dim = numpy.array([h,w])
	## Total number of bases
	kmax = (nmax+1)*(nmax+2)/2
	## Gauss-Laguerre scale
	beta = 1.0	###??
	
	## Oversampling factor
	osamp = 3	## integer
	ostep = 1.0/osamp
	odx = 0.5*(ostep-1.0)
	
	## Maximum radius
	rmax2 = w if w<h else h
	rmax2 *= 0.25*rmax2/(beta*beta)
	
##------------------------------------------------------------

	## Oversampled image size (pixels)
	oimsize = osamp*dim
	
	## Calculate some functions of space
	theta = numpy.zeros(oimsize)
	r2 = numpy.zeros(oimsize)
	expr2 = numpy.zeros(oimsize)
	X = numpy.arange(-0.5*oimsize[0],0.5*oimsize[0],1.0)
	Y = numpy.arange(-0.5*oimsize[1],0.5*oimsize[1],1.0)
	
	yw = odx-w/2-1.0
	for i,y in enumerate(Y):
		yw-=1.0
		xw = odx-h/2.-1.0
		for j,x in enumerate(X):
			xw-=1.0
			theta[i,j] = numpy.arctan2(yw,xw)
			r2[i,j] = xw*xw+yw*yw
			
	expr2 = numpy.exp(-r2*0.5/(beta*beta))
	print r2
	return 0
##------------------------------------------------------------

	## Initialise basis arrays
	obasis = numpy.zeros(oimsize)	## Oversampled
	basis = numpy.zeros([dim[0],dim[1],kmax])		## Original resolution basis
	
	## Shapelets
	k = 0
	for n in range (0,nmax+1):
		for m in range (n%2,n+1,2):
			
			hnmm=(n-m)/2
			## Compute ((n+m)/2)!/((n-m)/2)!
			fac = 1.0
			if n!=0:
				for p in range(hnmm>0,(n+m)/2+1): fac*=p
			fac = numpy.sqrt(1.0/(numpy.pi*fac))/beta			
			if hnmm%2: fac = -fac	###??
			
			## Oversampled basis image
			obasis = fac*eval_genlaguerre(r2, hnmm, m)*numpy.cos(m*theta)
			
			print eval_genlaguerre(r2, n, m)
			return 0
			#print obasis ###??
			
##------------------------------------------------------------
			## Pixellate to original resolution
			count = 0
			## Split up into blocks of size osamp
			for yblock in range (0,oimsize[0],osamp):
				for xblock in range (0,oimsize[1],osamp):
					
					## Add up the elements within each block
					for rownum in range (yblock, min(yblock+osamp,oimsize[0])):
						for colnum in range (xblock, min(xblock+osamp,oimsize[1])):
							basis[yblock/osamp,xblock/osamp,k] += obasis[rownum,colnum]
			
			k += 1
##------------------------------------------------------------
	
	## Normalise
	basis *= ostep*ostep
	
	return basis

##============================================================

"""Turns a stack of arrays into images."""

def arraytoimage(arraystack):
		
	if len(arraystack.shape)<3:
		print "Shapelet.py: arraytoimage: not a stack."
		return
	
	else:		
		for k in range (arraystack.shape[2]):
			basim = arraystack[:,:,k]		
			imag = Image.new("L",basim.shape)			
			basim = list(basim.flatten())			
			imag.putdata(basim)
			imag.save("test.png")
		
	return

##============================================================
if __name__=="__main__":
	main(int(argv[1]))
