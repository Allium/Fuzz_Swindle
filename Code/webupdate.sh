#!/bin/bash

## Code files
\cp ~/Dropbox/Hogg_2012/Code/HumVI/humvi/wherry.py ~/Dropbox/Hogg_2012/Code/PSF_Generator.py ~/Dropbox/Hogg_2012/Code/ImageScript.py ~/public_html/PSF_Eyeball/Code/

\rm ~/Dropbox/Hogg_2012/Code/HumVI/*.pyc ~/Dropbox/Hogg_2012/Code/HumVI/humvi/*.pyc
\cp -r ~/Dropbox/Hogg_2012/Code/HumVI/* ~/public_html/PSF_Eyeball/Code/HumVI_CS/


## Deconvolution
\cp ~/Dropbox/Hogg_2012/Data/Deconvolved/CFHTLS_03_g_Deconvolved*.pdf ~/public_html/PSF_Eyeball/Deconvolution/

## Colour
#\cp ~/Dropbox/Hogg_2012/Code/HumVI/images/test/CFHTLS_03_lw.pdf ~/public_html/PSF_Eyeball/Colour/


## Notes
\cp ~/Dropbox/Hogg_2012/Notes/Report.pdf ~/Dropbox/Hogg_2012/Notes/Report.tex  ~/public_html/PSF_Eyeball/Notes
\cp ~/Dropbox/Hogg_2012/Notes/progress.pdf ~/public_html/PSF_Eyeball/Notes

echo webupdate.sh: copied files.
