#!/usr/bin/env python

##=============================================================

"""
Inputs:
- Source position
- Source redshift
- Lens position
- Lens redshift
- Velocity dispersion
- External parameters (later)

Uses:
- Cosmology
- Density model

Outputs:
- Image positions

Program:
- Einstein radius
- Magnifications
- Image positions
- Cusp geometry
"""

##=============================================================

import numpy, pylab, time
from Distance import angular_diameter_distance as Da
from sys import argv
argc = len(argv)

##-------------------------------------------------------------
## Global constants
c=1.0
G=1.0

##-------------------------------------------------------------


def main():

  z1,z2 = 0.5,2.0
  M = 100
  x = 0.1
  
  tE = Einstein_angle(z1,z2,M)
  
  theta1 = 0.5*(x+sqrt(x*x+4*tE*tE))
  theta2 = 0.5*(x-sqrt(x*x+4*tE*tE))

  pylab.plot(x,0,"go",theta1,0,"bx",theta2,0,"bx")
  pylab.show()
  return None

def crit_surf_density(z1,z2):
  return c*c/(4*pi*G)*Da(z2)/(Da(z1)*Da(z1,z2))

def Einstein_angle(z1,z2,M):
  return numpy.sqrt(4*G*M*Da(z1,z2)/(c*c*Da(z2)*Da(z1)))


##=============================================================
if __name__=="__main__":
  main()
