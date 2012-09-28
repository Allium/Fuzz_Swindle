#!/usr/bin/env/python
##============================================================

"""
Inputs:
-------
1. Mode selection:
	a psf
	b kern
	c imdec
2b. Kernel size, e.g. [5,5]
2c. Image file
3c. Kernel file
"""

##============================================================

import numpy
import scipy
import os
import pyfits
import Image
import time
from scipy.signal import fftconvolve as convolve
from ImageScript import linear_rescale, stretch_params
from ImageScript import fits_pix, png_pix, pngcropwhite, writearray
from OSScript import file_seek
from common import lineno
from sys import argv

##============================================================

## Global variables
from common import cutoff

##============================================================
##============================================================
## Decide on program function.
## argv[1] is option, after that is parameters

def main(argv):
"""
	NAME
	  PSF_Generator.py

	PURPOSE
		1. Uses a PSF image to construct an "ideal" 2D Gaussian PSF
		2. Finds convolution kernel which maps between the "true" and "ideal" PSFs
		3. Deconvolves an entire image with this kernel

	COMMENTS
		- Uses standard deviation to characterise Gaussian rather than FWHM;
		f=sqrt(2ln2)*s.
	  
	USAGE
		python PSF_Generator.py [flags] function args
	
	FUNCTION
		1. psf
		2. kern
		3. imdec
	
	ARGS
		1. None
		2. Kernel-size
			 [9,9]
		3. Image-file  kernel-file
		(See examples)

	FLAGS
	  -h            Print this message

	INPUTS
		Depends on function selected		

	OPTIONAL INPUTS

	OUTPUTS
	  stdout        Useful information

	EXAMPLES
		
	                    
	BUGS
		- There is some disagreement between my convolution method and SciPy's.
		- Can't save directly to PNG in GPSF() -- save as FITS then convert.

	HISTORY
	  2012-06-12 started Sandford (NYU)
	"""
	
	## -------------------------------------------------------------------
	## Options and help-string
	
	try:
		opts, args = getopt.getopt(argv, "h",["help"])
	except getopt.GetoptError, err:
		## Print help information and exit:
		print str(err)
		print main.__doc__
		return
			   
	for o,a in opts:
		print o,a
		if o in ("-h", "--help"):
			print main.__doc__
			return
			
	## -------------------------------------------------------------------
	
	t_init = time.time()
	
	if args[1]=="psf":
		all_psfs(args)
	elif args[1]=="kern":
		## Find kernel dimensions from CLA -- e.g. "[5,5]"
		kdim=[int(x) for x in args[2][1:-1].split(",")]
		all_kernels(kdim)
	elif args[1]=="imdec":
		deconvolve_image(args[2],args[3])	
	
	return

##============================================================


##============================================================
##============================================================
## PSF Model
##============================================================
##============================================================

## Copy all the psf models in directory to Gaussians
def all_psfs(junk):	
	for fitsfile in file_seek("../Data/PSFs/","snap_*.fits"):
		GPSF(fitsfile)
	return

##============================================================

## Create a Gaussian PSF like the original image

def GPSF(filename):
	print filename
	## Get 2D Gaussian data
	Gdat = Gauss_2D(*moments(fits_pix(filename)))
	"""## Rescale to make things visible
	smin,smax = stretch_params(Gdat)[0],stretch_params(Gdat)[1]
	Gdat = linear_rescale(Gdat, smin,smax)"""
	##		Horrible method: write fits then convert to png
	if os.path.exists("temp.fits"): os.remove("temp.fits")
	hdu = pyfits.PrimaryHDU(Gdat)
	hdu.writeto("temp.fits")
	if zoom==True:
		outfile = os.path.splitext(filename)[0]+"_Gauss_z8.png"
		os.system("ds9 temp.fits -colorbar no -minmax -zoom 8 -saveimage png "+outfile+" -exit")
	else:
		outfile = os.path.splitext(filename)[0]+"_Gauss.png"
		os.system("ds9 temp.fits -colorbar no -minmax -saveimage png "+outfile+" -exit")
	os.remove("temp.fits")
	return
	
##============================================================
## From data, extract parameters needed to make a naive Gaussian fit
	## Argument data is 2D array

def moments(data):
	dim = data.shape
	L = max(dim)
	if dim[0]!=dim[1]: print "PSF_Generator.py: moments: image not square. Continuing."
	total = data.sum()
	X,Y = numpy.indices(dim)
	m_x = (X*data).sum()/total
	m_y = (Y*data).sum()/total
	ccol = data[:, int(m_y)]
	crow = data[int(m_x), :]
	width_x = numpy.sqrt(abs((numpy.arange(dim[1])-m_y)**2*ccol).sum()/ccol.sum())
	width_y = numpy.sqrt(abs((numpy.arange(dim[0])-m_x)**2*crow).sum()/crow.sum())
	return L, width_x, width_y, m_x, m_y, total
  
##============================================================
## Generate 2D Gaussian array

def Gauss_2D(size, w_x, w_y, x0, y0, scale=1.0):
	## Create mesh
	x=numpy.arange(0.0,size,1.0,float)
	y=x[:,numpy.newaxis]
	## Compute Gaussian
	gaussian=numpy.exp(-0.5*(((x-x0)/w_x)**2+((y-y0)/w_y)**2))	
	## Cut out pixels below threshold
	gaussian[gaussian<7.0e-6]=0.0
	## Normalise
	gaussian/=gaussian.sum()
	## Make same scale as original image	
	return scale*gaussian
	
	
##============================================================	


##============================================================
##============================================================
## Mapping kernel
##============================================================
##============================================================


## Once all GPSFs are calulated, work out kernels to and from
def all_kernels(kdim, writefile=True,makeimage=False):
		
	## Get files
	allpngs = numpy.sort(file_seek("../Data/GPSF/","snap*d0_CFHTLS_03_g*.png"))
	
	## Collect pairs of files
	for i in range (0,len(allpngs),2):
	
		## Extract image info
		image1 = png_pix(allpngs[i])
		image2 = png_pix(allpngs[i+1])
		
		## Cut out extraneous white pixels
		image1 = pngcropwhite(image1)
		image2 = pngcropwhite(image2)
		
		## Deconvolve
		A=get_kernel(image1,image2, kdim)
		B=get_kernel(image2,image1, kdim)	

##------------------------------------------------------------
			
		## Write kernels to file and make images
		
		## Gaussian to PSF
		outfile=os.path.splitext(allpngs[i])[0][:-4]+"_GausstoPSF_"+str(A.shape[0])+".krn"
		f=open(outfile,"w")
		f.write("## Convolution kernel taking 2D Gaussian to PSFEx image\n\n\n")
		writearray(f,A,True)
		print "PSF_Generator: kernel written to",outfile
		
		## PSF to Gaussian		
		outfile=os.path.splitext(allpngs[i])[0][:-4]+"_PSFtoGauss_"+str(B.shape[0])+".krn"
		f=open(outfile,"w")
		f.write("## Convolution kernel taking PSFEx image to 2D Gaussian\n\n\n")
		writearray(f,B,True)
		print "PSF_Generator: kernel written to",outfile
		
		print "\n"

##------------------------------------------------------------	
	
	return

##============================================================	

## Find convolution kernel whch maps image1 to image2 (both arrays)
	## Kernel must be odd and should be square
	## Some information from image2 gets shaved off
def get_kernel(image1, image2, kernel_dim):
	
##------------------------------------------------------------		

	## Enforce valid kernel shape
	if kernel_dim[1]!=kernel_dim[0]:
		kernel_dim[1] = kernel_dim[0]
		print "PSF_Generator.py: get_kernel: oblong kernel. Using",kernel_dim,"instead."
	if kernel_dim[0]%2==0:
		kernel_dim = [x-1 for x in kernel_dim]	## Unnecessarily Pythonic
		print "PSF_Generator.py: get_kernel: even kernel. Using",kernel_dim,"instead."
	
##------------------------------------------------------------		
		
	## Rearrange original image pixels into a suitable matrix 
	orig = stacker(image1, *kernel_dim)
	
	## Shave convolved image (->processed)
	desired_dim = numpy.array(image1.shape)-numpy.array(kernel_dim)+1
	proc = shave(desired_dim, image2).flatten()
	
	t0=time.time()	
	## Compute kernel
	kernel = LSsolve(orig,proc)
	print "PSF_Generator: get_kernel: kernel computation took",round(time.time()-t0,2),"seconds."
	
	## Normalise and shape
	kernel[kernel<cutoff] = 0.0
	kernel /= kernel.sum()
	kernel = numpy.reshape(kernel, kernel_dim)
	
	return kernel
	
##============================================================
## Stacks an array of pixel values into a matrix for dotting with flattened kernel
	## Arguments are image array and kernel shape (flipped)
	
def stacker(img_arr, krn_h,krn_w):
	
	## Dimensions
	h, w = img_arr.shape
	numcols = krn_h*krn_w
	numrows = (img_arr.shape[0]-krn_h+1)*(img_arr.shape[1]-krn_w+1)	
	
	## Prepare result array
	stacked = numpy.zeros([numrows, numcols])
	
	## Loop over rows (i) in the new array
	i,j,k=0,0,0
	while k<numrows:
		## Each row in new array is a kernel-shaped slice from old
		stacked[k,:] = img_arr[j:j+krn_w,i:i+krn_h].flatten()
		i+=1
		k+=1
		## Move down a row in original and start from column 0
		if i+krn_w-1==w:
			i=0
			j+=1
			continue
		
	return stacked
	
##============================================================
## Delete rows and columns from edges of an array
## so that it has shape = eudim.
def shave(eudim, arr):
	if ((numpy.array(arr.shape)-eudim)%2).all()!=0:
		print "shave: odd shave -> over-cut."
	while arr.shape[0]>eudim[0]:
		arr=numpy.delete(arr, 0, 0)	
		arr=numpy.delete(arr, -1, 0)	
	while arr.shape[1]>eudim[1]:
		arr=numpy.delete(arr, 0, 1)	
		arr=numpy.delete(arr, -1, 1)
	return arr

##============================================================
## Marix equation solvers

## Minimising square error via matrix multiplication
	## Seems to be "unstable", don't use
def puresolve(X,y):
	from numpy import dot
	t0=time.time()
	XT = numpy.transpose(X)
	ans = dot(dot(numpy.linalg.inv(dot(XT,X)),XT),y)
	print "puresolve time",round(time.time()-t0,5)
	return ans

## Solve system using QR decomposition	
def QRsolve(X,y):
	from numpy import dot
	t0=time.time()
	Q,R = numpy.linalg.qr(X)
	ans = dot(dot(numpy.linalg.inv(R),Q.T),y)
	print "QRsolve time",round(time.time()-t0,5)
	return ans

## Solve system using least squares		
def LSsolve(X,y):
	t0=time.time()
	kvec,resi,rank,sing = numpy.linalg.lstsq(X,y)
	#print "LSsolve time",round(time.time()-t0,3),"seconds"
	return kvec

## Not working	
def sympysolve(X,y):
	t0=time.time()
	X = sympy.Matrix(X)
	#ans= X.LDLsolve(y)
	ans= X.cholesky_solve(y)
	print "time",round(time.time()-t0,5)
	return ans

##============================================================

## 



##============================================================
##============================================================
## Reconvolution - check
##============================================================
##============================================================
## Convolves image1 (=filename) with kernel and compares with image2
def reconvolve(image1, image2, kernel):
	
	## Read in images & kernel and process into shape
	arr1 = pngcropwhite(png_pix(image1))
	kernel = numpy.loadtxt(kernel)
	newdim = numpy.array(arr1.shape)-numpy.array(kernel.shape)+1
	arr2 = shave(newdim, pngcropwhite(png_pix(image2)))
	
	## Reconvolved image -- should be integer array
	recon = numpy.around(convolve(arr1, kernel, mode="valid"))
	
	## Residue should be 0
	residue = recon-arr2
	
	## Visualise residue
	from pylab import imshow, show, colorbar
	imshow(residue)
	#colorbar()
	show()
	
	return

##============================================================
##============================================================
## Total Image Deconvolution
##============================================================
##============================================================
"""
Using the kernels for transforming PSF to Gaussian, deconvolve an entire image to a constant PSF
"""

##============================================================

def deconvolve_image(imagefile, kernelfile):
	
##------------------------------------------------------------
	## Read in image and kernel
	
	## Determine image file type and get pixels
	imgext = os.path.splitext(imagefile)[1]
	if imgext==".fits":	imgarr = fits_pix(imagefile)
	elif imgext==".png": imgarr = png_pix(imagefile) ## Doesn't work
	
	## For some reason the image is flipped at this point, so un-flip
	imgarr = imgarr[::-1,:]
	
	## Filter out noise
	imgarr[imgarr<cutoff] = 0.0
		
	## Ensure the right kernel file has been selected
	if "PSFtoGauss" in kernelfile:
		kernel = numpy.loadtxt(kernelfile)
	else:
		print "PSF_Generator.py: deconvolve_image: wrong kernel file."
		return
	kernel_h,kernel_w = kernel.shape
		
	# Should also check match with imagefile #
	
##------------------------------------------------------------	
	## Compute linalg objects from images
	
	## Honest dimensions for scene
	scene_dim = numpy.array(imgarr.shape)-numpy.array(kernel.shape)+1
	scene_siz = scene_dim[0]*scene_dim[1]
	
##------------------------------------------------------------
	
	convmeth = "scipy"
	
	## Manual convolution
	if convmeth=="manual":	
	
		## 1D array for SCENE (convolved with Gaussian)
		g_sc = numpy.empty(imgarr.size)#scene_siz
		## 1D array for IMAGE
		stride = imgarr.shape[0]
		imgarr = imgarr.flatten()

	##------------------------------------------------------------
	## Manual matrix product
		
		## Initialise kernel "vector"
		len_krn_lin = (stride)*(kernel_h-1)+kernel_w	## Still keeps a lot of zeros
		krn_lin = numpy.zeros(len_krn_lin)
		## Loop over slices in the kernel image
		for j in range (kernel_h):
			startcol = j*stride
			krn_lin[startcol:startcol+kernel_w] = kernel[j,:]
			
		t0 = time.time()
		## Perform linalg product
			## i labels the scene pixel and the slice in the original
		for i in range (scene_siz):
			imageslice = imgarr[i:i+len_krn_lin]
			g_sc[i] = numpy.dot(krn_lin,imageslice)
		print "PSF_Generator.py: deconvolve_image: vector multiplication took",\
					round(time.time()-t0,2),"seconds."
		
		## Delete spurious elements (from overlapping)
		i=len(g_sc)
		while i>=0:
			if i%stride+kernel_w > stride:
				g_sc = numpy.delete(g_sc,i)
			i-=1
		## Delete spurious elements from declaring it too big
		g_sc = numpy.delete(g_sc,slice(scene_siz-len(g_sc)-1,-1))
		if scene_siz-len(g_sc): print "LINE #"+str(lineno())+": size discrepancy"


##------------------------------------------------------------
	
	elif convmeth=="scipy":
		t0=time.time()
		## Do convolution using scipy
		imgarr = numpy.array(imgarr, dtype="float64")
		g_sc = convolve(imgarr, kernel, mode="valid")
		print "PSF_Generator.py: deconvolve_image: SciPy convolution took",\
					round(time.time()-t0,2),"seconds."
		
	else:
		print "LINE #"+str(lineno())+": convmeth error, abort."
		return
		
##------------------------------------------------------------
	
	## Reshape
	if g_sc.shape[0]!=scene_dim[0] or g_sc.shape[1]!=scene_dim[1]:
		print "PSF_Generator.py: deconvolve_image: reshaping."
		try:
			"""
			## Manual reshape
			new = numpy.empty(scene_dim)
			for i in range (len(g_sc)/scene_dim[0]):
				new[i,:] = g_sc[i*scene_dim[0]:(i+1)*scene_dim[0]]
			g_sc = new
			del(new)"""
			g_sc = g_sc.reshape(scene_dim)
		except ValueError:	
			print "PSF_Generator.py: deconvolve_image: output has wrong shape. Investigate."

##------------------------------------------------------------
	
	## Filter out (computer) noise
	g_sc[g_sc<cutoff] = 0.0
	
	## Outfile name
	imagename = os.path.basename(imagefile)
	info = imagename[imagename.find("CFHT"):imagename.find("sci")+3]
	outfile = "../Data/Deconvolved/gsc_"+info+".png"
	
	## Rescale
	if 1:
		## Scaling parameters
		vmin,vmax = stretch_params(g_sc)
		## Stretch
		g_sc = linear_rescale(g_sc, vmin, vmax*0.5)
		## Modify filename
		outfile = outfile[:-4]+"_rescaledl.png"
	
	## Save image
	scipy.misc.imsave(outfile,g_sc)
	print "PSF_Generator.py: deconvolve_image: image saved to",outfile
	
	return None
	
	
	
##============================================================	
"""
##============================================================
## Generate linear-algebra convolution kernel from kernel image
	#	krn_mat = kernel_matrix(kernel, imgarr.shape, len(g_sc))
	## Note this is memory-inefficient: result is sparse, upper-triangular
	## matrix, but I store all elements.
	## Actually, this turns out to be prohibitively inefficient
def kernel_matrix(kernel, image_shape, scene_length):
	
	## Dimensions
	stride = image_shape[0]
	image_length = image_shape[0]*image_shape[1]
	kernel_h,kernel_w = kernel.shape
	
	## Initialise kernel matrix: image_length columns
		## Should have scene_length rows, but overassign for now
	print image_length, scene_length
	krn_mat = numpy.zeros([scene_length, image_length])
	
	## Stretch out kernel image; we plug this straight into the kernel matrix
	len_krn_lin = (stride+1)*(kernel_h-1)+1
	krn_lin = numpy.zeros(len_krn_lin)
	## Loop over slices in the kernel image
	for j in range (kernel_h):
		startcol = j*stride
		krn_lin[startcol:startcol+kernel_w] = kernel[j,:]
	
	## Loop over rows in the kernel matrix
	for i in range (image_length):
		try:
			if i%stride+kernel_w <= stride:
				krn_mat[i,i:i+len_krn_lin] = krn_lin
		except ValueError:	## Invoked when end is reached
			print "PSF_Generator: kernel_matrix: reached end at row",i
			break
	
	## Delete all rows which sum to zero
	krn_mat = krn_mat[krn_mat.sum(1)!=0]
	## Make sure the kernel matrix has the right number of rows
	if krn_mat.shape[0]!=scene_length:
		print "PSF_Generator: kernel_matrix: result is nonconformable matrix."
		
	return krn_mat
"""




##============================================================
if __name__=="__main__":
	main(argv)
