#!/usr/bin/perl -w
# ======================================================================
#+
$usage = "

NAME
        gallery.pl

PURPOSE
        Makes an (m x n) gallery of images.

USAGE
        gallery [flags] [options] file*

FLAGS
        -u      Print this message
        -A4     Use A4 paper (def=USletter)
        -fill   No margins! Good for papers
        -pack   Close packing of sub-figures
        -pdf    Make a pdf file using pdflatex - can take jpg input!

INPUTS
        file*            List of images

OPTIONAL INPUTS
        -o      file     Write output to \"file\" (def=gallery.[ps,pdf])
        -r      i        Image rotation angle (0,90,180,270 degrees, def=0)
        -x      i        No. of panels in x direction
        -y      i        No. of panels in y direction
        -t               Title (filename) is added as a caption
        -s               Shorten title by only taking first part before '_'
        -v      i        Verbosity (0, 1 or 2, default=0)
        -k               Keep report from [pdf]latex [and dvips]
        -p               Number the pages of the gallery

OUTPUTS
        gallery.[ps,pdf] or explicitly named output file

OPTIONAL OUTPUTS

COMMENTS

EXAMPLES

BUGS
  - Little useful output to the screen
  - No way to recover from latex error except ctrl-z ; kill
  - Sensitive to too many images?

REVISION HISTORY:
  2005-08-15  Started Marshall (KIPAC)

\n";
#-
# ======================================================================

#$sdir = $ENV{'SCRIPTUTILS_DIR'};
#require($sdir."/perl/esystem.pl");	##My changes
require("esystem.pl");
$doproc = 1;

# Sort out options:

use Getopt::Long;
GetOptions("o=s", \$outfile,
           "x=i", \$m,
           "y=i", \$n,
           "r=i", \$theta,
           "A4", \$useA4,
           "v=i", \$verb,
           "t", \$titlise,
           "s=i", \$titlelength,
           "fill", \$fill,
           "pack", \$squash,
           "pdf", \$pdf,
           "k", \$keepreport,
           "p", \$pnums,
           "u", \$help
           );

(defined($help)) and die "$usage\n";
$num=@ARGV;
($num>0) or die "$usage\n";

(defined($verb)) or ($verb = 0);

(defined($pdf)) or ($pdf = 0);
if ($pdf) {
  (defined($outfile)) or ($outfile="gallery.pdf");
} else {
  (defined($outfile)) or ($outfile="gallery.ps");
}

(defined($m)) or ($m = 2);
(defined($n)) or ($n = 2);
(defined($theta)) or ($theta = 0);
(($theta==0) or ($theta==90) or ($theta==180) or ($theta==270)) or die "$usage\n";

# Count postscript files, and read filenames into an array:

$nfiles = 0;
while (defined($x = shift)){
    $file[$nfiles] = $x;
    $nfiles++;
}
($nfiles == 0) and die "ERROR: No files supplied.\n\n$usage\n";
# ($nfiles > ($m*$n)) and die "ERROR: too many files.\n\n$usage\n";

# Guess number of pages:
$npages = $nfiles/($m*$n);

($verb>0) and print STDOUT "\nEstimated no. of pages = $npages\n";

# Title formatting:
if (defined($titlise)){
  (defined($titlelength)) or ($titlelength = 100);
} else { 
  $titlelength = 0;
}  

# Write preamble of latex file:

($verb>0) and print STDOUT "\nWriting latex file...\n";

$root = "gallery";
$texfile = $root.".tex";
$auxfile = $root.".aux";
$logfile = $root.".log";
$dvifile = $root.".dvi";
$reportfile = $root.".report";
open (TEX, ">$texfile") or die "Couldn't open $texfile: $!";

# Page definition

if (defined($useA4)) {

print TEX "\\documentclass[11pt,a4paper,twoside]{article}\n";
print TEX "\\setlength\\paperheight{29.7cm}\n";
print TEX "\\setlength\\paperwidth{21.0cm}\n";
print TEX "\\setlength\\voffset{-2.54cm}\n";
print TEX "\\setlength\\hoffset{-2.54cm}\n";

} else {

print TEX "\\documentclass[11pt,letterpaper,twoside]{article}\n";
print TEX "\\setlength\\paperheight{11in}\n";
print TEX "\\setlength\\paperwidth{8.5in}\n";
print TEX "\\setlength\\voffset{-1in}\n";
print TEX "\\setlength\\hoffset{-1in}\n";

}

# Margins:

if (defined($fill)){

print TEX "\\setlength\\oddsidemargin{0.0in}\n";
print TEX "\\setlength\\evensidemargin{0.0in}\n";
# print TEX "\\setlength\\topmargin{-0.5in}\n";
print TEX "\\setlength\\topmargin{1in}\n";
print TEX "\\setlength\\textwidth{\\paperwidth}\n";
print TEX "\\setlength\\textheight{\\paperheight}\n";

} elsif (defined($useA4) ){

print TEX "\\setlength\\oddsidemargin{2.0cm}\n";
print TEX "\\setlength\\evensidemargin{2.0cm}\n";
print TEX "\\setlength\\topmargin{2.0cm}\n";
print TEX "\\setlength\\textwidth{17.0cm}\n";
print TEX "\\setlength\\textheight{25.7cm}\n";

} else {

print TEX "\\setlength\\oddsidemargin{0.75in}\n";
print TEX "\\setlength\\evensidemargin{0.75in}\n";
# print TEX "\\setlength\\topmargin{1.5in}\n";
print TEX "\\setlength\\topmargin{0.5in}\n";
print TEX "\\setlength\\textwidth{7.0in}\n";
print TEX "\\setlength\\textheight{10.25in}\n";

}

print TEX "\\usepackage{epsfig}\n";
print TEX "\\begin{document}\n";

if (defined($pnums)){
  print TEX "\\pagestyle{plain}\n";
} else {
  print TEX "\\pagestyle{empty}\n";
}

# Calculate minipage width and orientation:

if (defined($squash)){
  $squash = 1;
  $pad = 0.01;
}else{
  $squash = 0;
  $pad = 0.02;
}  
$dx = 1.0/$m - $pad;
if  (($theta==0) or ($theta==180)) {
  $y = "width";
} else {
  $y = "height";
}

# npages loop over pages:

$k=0;

for ($l=0; $l<$npages; $l++){

  # m x n loop over filenames within a page:

  for ($j=0; $j<$n; $j++){

    print TEX "\n\\begin{figure}[!ht]\n";

    for ($i=0; $i<$m; $i++){

      if ($k < $nfiles){
        $filenotfound = 0;
        open (TEST, "$file[$k]") or $filenotfound = 1;
        close(TEST);
        ($filenotfound) and print "Warning: $file[$k] does not exist.\n";
      }  

      if ($filenotfound or $k >= $nfiles){
        if ($pdf) {
#           print "ERROR: cannot make blank images in pdf mode.\n";
#           exit;
          MakeBlankJPG();
          $file[$k] = "blank.jpg";
        } else {
          MakeBlankPS($file[0]);
          $file[$k] = "blank.ps";
        }  
        $shortname = "";
      } else {
#       Short filename for captions: 
        $shortname = $file[$k];
        $shortname =~ s/.*\/(.*)/$1/;
        if ($titlelength >= 0) {
          $shortname = substr $shortname, 0, $titlelength;
        } else {
          @values = split('_',$shortname);
          $shortname = $values[0];
        }
      }  
      
#  Check extension: 
      $extension = $file[$k];
      $extension =~ s/.*\.(.*)/$1/;
      if ($pdf and ($extension eq "ps" or $extension eq "eps")){
        print "ERROR: cannot use postscript images in pdf mode ($file[$k]).\n";
        exit;
      } elsif (not $pdf and ($extension ne "ps" and $extension ne "eps")){
        print "ERROR: can only use postscript images in ps mode ($file[$k]).\n";
        exit;
      }

      print TEX "\\begin{minipage}[b]{$dx\\linewidth}\n";
      print TEX "\\centering\\epsfig{file=$file[$k],$y=\\linewidth,angle=$theta,clip=}\n";
      ($titlise) and print TEX "\\footnotesize\\verb=$shortname=\n";

      print TEX "\\end{minipage} \\hfill\n";

      $k++

    }

    print TEX "\\end{figure}\n";
#     ($titlise) and print TEX "\\vspace{-\\baselineskip}\n";
    print TEX "\\vspace{-\\baselineskip}\n";
    defined($fill) and print TEX "\\vspace{-\\baselineskip}\n";

  }
  print TEX "\\clearpage\n";

}
print TEX "\\end{document}\n";
close(TEX);

# Now do system calls to latex and dvips, and clean up:

if ($pdf) {
  ($verb>0) and print STDOUT "Running pdflatex...\n";
  &esystem("echo x | pdflatex $texfile >& $reportfile",$doproc,$verb);
  &esystem("mv gallery.pdf $outfile",$doproc,$verb);
} else {
  ($verb>0) and print STDOUT "Running latex...\n";
  &esystem("echo x | latex $texfile >& $reportfile",$doproc,$verb);
  ($verb>0) and print STDOUT "Running dvips...\n";
  &esystem("dvips -f -Pcmz $dvifile -o $outfile >> $reportfile 2>&1",$doproc,$verb);
}

# Clean up:
&esystem("rm -f $dvifile $auxfile $logfile junk",$doproc,$verb);


($keepreport) or &esystem("rm -f $texfile $reportfile blank.ps blank.jpg",$doproc,$verb);

if ($pdf) {
  ($verb>0) and print STDOUT "\nPDF written to $outfile\n";
} else {
  ($verb>0) and print STDOUT "\nPostscript written to $outfile\n";
}

($verb>0 and $keepreport) and print STDOUT "\nLaTeX report written to $reportfile";
($verb>0 and $keepreport) and print STDOUT "\nLaTeX retained in $texfile\n\n";

#END

#=======================================================================

sub MakeBlankPS{
    my ($template) = @_;

$bpsfile="blank.ps";
&esystem("echo '%!PS-Adobe-3.0' > $bpsfile",$doproc,0);
&esystem("grep -e BoundingBox $template | grep -v Page >> $bpsfile",$doproc,0);
&esystem("echo '%%Title: Blank postscript' >> $bpsfile",$doproc,0);
&esystem("echo '%%EOF' >> $bpsfile",$doproc,0);
}

#=======================================================================

sub MakeBlankJPG{

&esystem("convert -size 100x100 xc:white blank.jpg",$doproc,0);
}

#=======================================================================
