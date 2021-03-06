import os
import copy
import numpy as np
import pylab as pl

import astropy.io.fits   as fits
import matplotlib.pyplot as plt

from   utils import latexify


latexify(columns=2, equal=False, fontsize=11, ratio=0.5, ggplot=True, usetex=True)

root = os.environ['CSCRATCH']
dat  = fits.open(root + '/COSMOS/COSMOS15/cat.fits')

##  Area flags:
##  --  FLAG_COSMOS, 2 deg2.
##  --  FLAG_DEEP  
##

##  Photo-z flags.
##  --  ZQ,  Photo-z with AGN library. 
##  --  ZP_2,  2nd best photo-z
##  --  PHOTOZ.
##

##  Expectation for Ks < 24.7 in Ultra Deep: 1.5e5.
##                                     Deep: 0.9e5.

INCOSMOS   =  dat[1].data['FLAG_COSMOS']   ##  1: 2.0 deg2 COSMOS area
INDEEP     =  dat[1].data['FLAG_DEEP']     ##  1: Ultra-deep stripes, 0: deep stripes.
INSHALLOW  =  dat[1].data['FLAG_SHALLOW']

Ks         =  dat[1].data['Ksw_MAG_APER3']
Klim       =  Ks < 24.7

ips        =  dat[1].data['ip_MAG_APER3']  ##  3'' aperture. 

## 
zs         =  dat[1].data['PHOTOZ']        ##  Median of the photo-z likelihood.  

cmap       =  plt.get_cmap("tab10")

##  [INCOSMOS, INDEEP], [2., 0.46], ['s', '^']
for X, AREA, marker in zip([INDEEP], [0.46], ['^']):
 for ii, ilim in enumerate(np.arange(23.6, 26.1, 0.5)):  
  iplim      =  ips < ilim

  ##  Photo-z is not NAN and AREA \in [COSMOS, INDEEP].
  cut        =  np.logical_and(X, np.invert(np.isnan(zs)))
  cut        =  np.logical_and(cut, iplim)

  good_zs    =  zs[cut]

  dz         =  0.25
  bins       =  np.arange(0.0, 6.0, dz)

  hist, bins =  np.histogram(good_zs, bins)
  midz       = (np.roll(bins, -1) + bins) / 2.
  midz       =  midz[:-1]

  if AREA < 0.5:
   label=r"$i' < %.1lf$" % ilim
  
  else:
   label=""

  pl.semilogy(midz, hist / AREA, markersize=5, label=label, marker=marker, c = cmap(ii), alpha=0.8)

  print('%.3lf \t %d' % (ilim, np.sum(cut)))

pl.xlim(0.0,   4.5)
pl.ylim(1.e2, 5.e4)

pl.xlabel(r'$z$', fontsize=13)
pl.ylabel(r'$N$ per sq. deg. per $\Delta z$=%.2lf' % dz)

pl.legend(ncol=2, frameon=False)

plt.tight_layout()

##  pl.show()
pl.savefig('plots/cosmos.pdf')
