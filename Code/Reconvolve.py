#!usr/bin/env/python
##============================================================

"""
Takes PNG images and convolves them
"""
##============================================================

import numpy
import Image
from scipy.signal import convolve
from sys import argv

##============================================================
def main(im1,im2):
	
	image1="face.png"
	image2="ds9.png"
	
	n=0

	I1=Image.open(image1)
	I2=Image.open(image2)
	
	x1,y1=I1.size
	x2,y2=I2.size
	
	data1=numpy.reshape(numpy.array(I1.getdata()),[x1,y1,3])
	data2=numpy.reshape(numpy.array(I2.getdata()),[x2,y2,3])
	
	conv=convolve(data1,data2,mode="same")
		
	imag=Image.new("RGB",conv.shape[:2])
	
	print data1.shape, data2.shape, conv.shape
	
	conv=list(conv.flatten())
	
	imag.putdata(conv,1.0,0.0)
	imag.show()
	
	return


##============================================================
if __name__=="__main__":
	main(argv[1],argv[2])
