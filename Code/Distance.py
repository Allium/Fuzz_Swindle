"""
Inputs:
- Redshifts
- Cosmology (eventually)

Outputs:
- Cosmological distances
"""

##=============================================================

import numpy, scipy
from numpy import sqrt
from sys import argv

##-------------------------------------------------------------
## Global constants

## Speed of light
c = 299792458.
## Cosmology parameters (Om, OL, h, w0, wa)
cosm = [0.26, 0.74, 0.72, -1.06, 0.0]
O_m = cosm[0]
O_L = cosm[1]
O_k = 1.0 - cosm[0] - cosm[1]
h = cosm[2]
w0 = cosm[3]
wa = cosm[4]

##=============================================================

## Input redshifts (order doesn't matter)
z1 = argv[1]
z2 = argv[2]

##=============================================================

## Normalised Hubble parameter (NHP) at redshift z
def NHP(z):
  return (O_m*(1.+z)**3. + O_k*(1.+z)**2 + O_L*(1.+z)**(3.*(1.+w0+wa))*numpy.exp(-3*wa*z/(1.+z)))**(0.5)

##-------------------------------------------------------------

## Age of the Universe at redshift z
def age(z):
  f = lambda x: 1.0/( NHP(x)*(1.+x) )
  result, error = scipy.integrate.quad(f, z, numpy.inf)
  return result * (10./h)

##-------------------------------------------------------------

## The comoving distance between events at z1 and z2
  ## Often denoted as chi
def comoving_distance(z1,z2=0.):
  if z2<z1: z1,z2 = z2,z1
  f = lambda z: 1.0/NHP(z)
  return (c/(h*1e5))*integrate.romberg(f, z1, z2)

##-------------------------------------------------------------

## Transverse comoving distance
  ## Often denoted r(chi)
def D_ct(z1,z2=0.):
  dc = comoving_distance(z1,z2)*(1e5*h/c)
  ## Transverse distance calculation depends on curvature
  if O_k == 0.0:
    dtc = dc
  elif O_k>0.0:
    dtc = numpy.sinh(sqrt(O_k)*dc)/sqrt(O_k)
  else:
    dtc = numpy.sin(sqrt(-1.*O_k)*dc)/sqrt(-1.*O_k)
  ## Dimensions
  return (c/(h*1e5))*dtc

##-------------------------------------------------------------

## Angular diameter distance (from Euclidean geometry)
def angular_diameter_distance(z1,z2=0.):
  if z2<z1: z1,z2 = z2,z1
  return D_ct(z1,z2)/(1.+z2)

##-------------------------------------------------------------

## Luminosity distance (from flux considerations)
def luminosity_distance(z):
  return (1.+z)*D_ct(z)

##-------------------------------------------------------------

## Critical density

##-------------------------------------------------------------

## Time-delay distance (useful for lensing)
def time_delay_distance(z1, z2):
  return D_ct(0.,z1) * D_ct(0.,z2) / D_ct(z1, z2)
