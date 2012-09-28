#!/usr/bin/env/python
##============================================================

import os
from sys import argv
from numpy import sort
from shutil import move
from OSScript import file_seek

##============================================================

def main():
	if argv[1]=="cmap": cmap()
	elif argv[1]=="ispr": ispr()
	elif argv[1]=="psfs": psfs()
	elif argv[1]=="manual": manual(*argv)
	else: print "GalleryScript.py: main: select an option."
	return

##============================================================
## Gallery of cmaps
def cmap():

	## File lists
	originals=sort(file_seek("../Data/", "CFHTLS_*_sci.png"))
	stretched=sort(file_seek("../Data/", "CFHTLS_*_rescaled.png"))
	brights=sort(file_seek("../Data/", "CFHTLS_*_br*.png"))
	BB=sort(file_seek("../Data/", "CFHTLS_*_cBB.png"))
	HE=sort(file_seek("../Data/", "CFHTLS_*_cHE.png"))
	
	## Check
	numfiles=len(originals)
	if len(brights)!=numfiles or len(BB)!=numfiles or len(HE)!=numfiles or len(stretched)!=numfiles:
		print "GalleryScript.py: cmap: There are gaps. Abort."
		print numfiles,len(brights),len(BB),len(HE),len(stretched)
		return
	
	## Make ordered list of files
	full_list=""
	for i in range (numfiles):
		full_list+=" "+originals[i]
		full_list+=" "+stretched[i]
		full_list+=" "+brights[i]
		#full_list+=" "+BB[i]
		full_list+=" "+HE[i]
	
	print "GalleryScript.py: cmap: files found. Proceed with latex."
	
	## Destroy current version
	if os.path.isfile("../Data/cmap.pdf"):	os.remove("../Data/cmap.pdf")	
	
	## Call gallery script
	os.system("perl gallery.pl "+full_list+" -pdf -t -o ../Data/cmap.pdf")
	
	## Rename gallery.pl output
	#os.rename("gallery.pdf","cmaps.pdf")
	## Move to appropriate location
	#move("cmaps.pdf","../Data/")
	
	return
	
##============================================================	
## Gallery showing side-by-side original image, stars, psf snapshot and residuals
def ispr():
	
	## File lists
	originals=sort(file_seek("../Data/", "CFHTLS_*_sci_rescaled.png"))
	stretched=sort(file_seek("../Data/", "CFHTLS_*_rescaled.png"))
	stars=sort(file_seek("../Data/", "samp_CFHTLS*_rescaled.png"))
	psf=sort(file_seek("../Data/", "snap_n1_d0_CFHTLS*_rescaled_z8.png"))
	residuals=sort(file_seek("../Data/", "resi_CFHTLS*_rescaled.png"))
	
	## Check
	numfiles=len(originals)
	if len(stars)!=numfiles or len(psf)!=numfiles or len(residuals)!=numfiles or len(stretched)!=numfiles:
		print "GalleryScript.py: ispr: There are gaps. Abort."
		print numfiles,len(stars),len(psf),len(residuals),len(stretched)
		return
	
	## Make ordered string of files
	full_list=""
	for i in range (numfiles):
		#full_list+=" "+originals[i]
		full_list+=" "+stretched[i]
		full_list+=" "+stars[i]
		full_list+=" "+psf[i]
		full_list+=" "+residuals[i]
	
	print "GalleryScript.py: ispr: files found. Proceed with latex."
	
	## Destroy current version
	if os.path.isfile("../Data/ispr.pdf"):	os.remove("../Data/ispr.pdf")	
	
	## Call gallery script
	os.system("perl gallery.pl "+full_list+" -pdf -t -o ../Data/ispr.pdf")
	
	return
	
##============================================================	
## Gallery showing psf kernels
def psfs():
	
	## File lists
	psf_files=sort(file_seek("../Data/", "*fit?.png"))
	
	## Check	
	
	## Make ordered string of files
	full_list=""
	for i in range (len(psf_files)):
		full_list+=" "+psf_files[i]
	
	print "GalleryScript.py: psfs: files found. Proceed with latex."
	
	## Destroy current version
	if os.path.isfile("../Data/psfs.pdf"):	os.remove("../Data/psfs.pdf")	
	
	## Call gallery script
	os.system("perl gallery.pl "+full_list+" -pdf -t -o ../Data/psfs.pdf")
	
	return

##============================================================	
## e.g. 

def manual(module,sel, directory, liststring1, liststring2, outname):
	
	## File lists
	files_1=sort(file_seek(directory, liststring1+".png"))
	files_2=sort(file_seek(directory, liststring2+".png"))
	
	## Check
	if len(files_1)!=len(files_2):
		print "GalleryScript.py: manual: there are gaps. Abort."
		return
	
	## Make ordered string of files
	full_list=""
	for i in range (len(files_1)):
		full_list+=" "+files_1[i]	
		full_list+=" "+files_2[i]
		
	print "GalleryScript.py: manual: files found. Proceed with latex."
	
	## Destroy current version
	if os.path.isfile("../Data/"+outname+".pdf"):	os.remove("../Data/"+outname+".pdf")	
	
	## Call gallery script
	os.system("perl gallery.pl "+full_list+" -pdf -t -o ../Data/"+outname+".pdf")
	
	return
	
##============================================================
if __name__=="__main__":
	main()
