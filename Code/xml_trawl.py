#!/usr/bin/env/python
##============================================================

import numpy
import os
import glob
from sys import argv

##============================================================

## Find xml files in any subdirectory of top_dir

def file_seek(top_dir):
	
	file_list = []
	
	## Walk through the directories, starting with top_dir
	for root,dirs,files in os.walk(top_dir):
		## Find .xml file
		for xml_file in glob.iglob( os.path.join( root, "*.xml" )):
			file_list += [xml_file]
	
	return file_list

##============================================================

## Read relevant parameters from an xml file
	## FWHM_Mean	22
	## Ellipticity_Mean	25
	## FWHM_pixelfree_Mean	34
	## Ellipticity_pixelfree_Mean	37

def retrieve_data():
	
	"""
	for each one
	loadtxt
	trawl
	find relevant parameters (default / given in input as kwarg)
	store in list or something
	
	return this list
	"""
	
	elements=[22,25]
	
	xml_files = file_seek(top_dir)
	
	for xml in xml_files:
		
		## Integer counter
		count = 0
		## Load xml file into big array
		DAT = numpy.loadtxt(xml)
		
		## Walk though elements of DAT
		for i in range(len(DAT)):
			## Find region of interest: the table
			if DAT[i:i+16]=="<DATA><TABLE>":
				for j in range([i+16,len(DAT)]):
					## Within the table, count the entry number
					if DAT[j:j+1]=="TD":
						count++
						## If count # marks an element of interest, store it
						for l in elements:
							if 2*count+1==l: j1=j
							elif
					
	
	return
	
##============================================================

def scrape_xml():

##============================================================

if __name__=="__main__":
	file_seek()
