#!/usr/bin/env/python
##============================================================
"""To be called from Code directory"""

"""
py SExScript.py ../Data/CFHTLS_03/CFHTLS_03_g_sci/CFHTLS_03_g_sci.fits

## Default run of PSFEx
py SExScript.py ../Data/CFHTLS_03/CFHTLS_03_g_sci/CFHTLS_03_g_sci.cat
## Make only one image in SNAPSHOTS (rather than grid)
py SExScript.py ../Data/CFHTLS_03/CFHTLS_03_g_sci/CFHTLS_03_g_sci.cat snap 1 2


"""

import os
import numpy
import time
from sys import argv
from OSScript import file_seek

##============================================================

## Seek out all fits image files in directory and make psfs
	## argv[1] is the directory to look at
	## argv[2] is option for PSFEx (maybe later)
	
def main():
	
	t0=time.time()
	
	fits_list = file_seek(argv[1], "CFHTLS_*sci.fits")
	
	for fitsfile in fits_list:
		## Run SExtractor
		SEx(["a",fitsfile])
		## Run PSFEx on fresh .cat
		PSFEx(["a",fitsfile[:-4]+"cat"])
	
	print "\nSExScript.py: analysed",len(fits_list),"fits images."
	print "Took",round(time.time()-t0,2),"seconds."
	return
	
##============================================================

## Runs SExtractor or PSFEx depending on input file
	## argv[1] is a FILE
def choose(argv):
	
	if argv[1][-4:]=="fits": SEx(argv)
	elif argv[1][-3:]=="cat": PSFEx(argv)
	else: print "SExScript: provide suitable input file."
	
	return

##============================================================

def SEx(imgname):
	
	## Remove junk at beginning
	imgname=imgname[1]
	
	## Check the input
	if imgname[-5:]!=".fits":
		print "SExScript.py: SEx: invalid input image."
		return
		
	## PSFEx needs output in FITS_LDAC format, and different data in parameter file
	confile = " -c prepsfex.sex "
	params  = " -PARAMETERS_NAME prepsfex.param "
	cattype = " -CATALOG_TYPE FITS_LDAC "
	catname = " -CATALOG_NAME "+imgname[:-5]+".cat "

	commands = "sex "+imgname+confile+params+cattype+catname
	#print "\n Commands: "+commands+"\n"
		
	## Execute commands
	os.system(commands)
	
	return

##============================================================

## Modify arguments of PSFEx
	## cat[1] should be a .cat file
	## cat[2+] are additional arguments for e.g. snap

def PSFEx(cat):

	## Check the input
	if cat[1][-4:]!=".cat":
		print "SExScript.py: PSFEx: invalid input catalogue."
		return
	
	## Input catalogue
	catname = os.path.basename(cat[1])[:-4]
	
	## Destination prefix
	p = cat[1][:-4]+"_Diagnostics/"
	if os.path.isdir(p)==False: os.mkdir(p)

##------------------------------------------------------------

	## Choice of output depends on argv[2]
	
	if len(cat)>2:
	
		## If we want to return only PSF snapshot(s)
		if cat[2]=="snap":
		
			## How many more arguments are we expecting (than default case)?
			moreargs = 3	## cat[2] is "snap"; cat[3] is number of snapshots; cat[4] is degree of fit
			
			## Only one check image: snap_*.fits
			imgdest = " -CHECKIMAGE_TYPE SNAPSHOTS  -CHECKIMAGE_NAME "+p+"snap_n"+cat[3]+"_d"+cat[4]+".fits "
			params  = " -PSFVAR_NSNAP "+cat[3]+"  -PSFVAR_DEGREES "+cat[4]+"  "
			## No check plots (png)
			pltdest = " -CHECKPLOT_TYPE NONE "
			## Destination for XML file (xml)
			xmldest = " -WRITE_XML N "
	
		elif cat[2]=="none":
				## How many more arguments are we expecting (than default case)?
				moreargs = 0			
				## Only one check image: snap_*.fits
				imgdest = " -CHECKIMAGE_TYPE NONE "
				## No check plots (png)
				pltdest = " -CHECKPLOT_TYPE NONE "
				## No XML file (xml)
				xmldest = " -WRITE_XML N "
				## Default 1-degree fit	
				params  = " -PSFVAR_DEGREES 1 "+" BASIS_TYPE PIXEL_AUTO
			
	## Default: compute everything
	else:
		## How many more arguments are we expecting?
		moreargs = 0	
		## Destination for check images (fits)
		imgdest = p+"chi.fits,"+p+"proto.fits,"+p+"samp.fits,"+p+"resi.fits,"\
							+p+"snap_n9_d2.fits,"+p+"moffat.fits,"+p+"submoffat.fits,"+p+"subsym.fits"
		imgdest = " -CHECKIMAGE_NAME "+imgdest
		## Destination for check plots (png)
		pltdest = p+"fwhm,"+p+"ellipticity,"+p+"counts,"+p+"countfrac,"+p+"chi2,"+p+"resi"
		pltdest = " -CHECKPLOT_NAME "+pltdest
		## Destination for XML file (xml)
		xmldest = " -XML_NAME "+p+catname+".xml "
		params  = " "
	
##------------------------------------------------------------
	
	## Sometimes want to specify further parameters in default.psfex
	## These are given in the last command-line argument (" ")
	if len(cat)>2+moreargs:
		params+=cat[2+moreargs]
	
	## Stick it all together
	commands = "psfex "+cat[1]+imgdest+pltdest+xmldest+params
	#print "\n"+commands+"\n"
	#return
	
	## Execute commands
	os.system(commands)
	
	return

##============================================================

## Generate snapshots of varying quality and coverage
	## Called from python line (use OSScript.file_seek first)
def snapall(catfile):
	for nsnap in [1,9]:
		for deg in [0,1,2,3]:
			PSFEx(["dummy",catfile,"snap",str(nsnap),str(deg)])
	return

	
##============================================================

if __name__=="__main__":
	main()
