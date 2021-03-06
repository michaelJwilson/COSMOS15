import os
import numpy               as      np
import pylab               as      pl 
import astropy.io.fits     as      fits
import matplotlib.pyplot   as      plt

from   HilderandtSN        import  merr, ferr
from   luptitudes          import  luptitude
from   utils               import  latexify
from   depths              import  get_depths
from   colourcut_dNdz      import  colourcut
from   app_mags            import  get_colors


##  Selection type. 
ttype   = 'g'
DODEPTH = 'Degraded'  ##  ['Full', 'Degraded']
plotit  =  True

##  Seed.
np.random.seed(seed=314)

##  Load COSMOS catalogue. 
root    = os.environ['CSCRATCH']
dat     = fits.open(root + '/COSMOS/COSMOS15/cat.fits')

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

umag       =  dat[1].data['u_MAG_APER3']
Bmag       =  dat[1].data['B_MAG_APER3']
Vmag       =  dat[1].data['V_MAG_APER3']
rmag       =  dat[1].data['r_MAG_APER3']
imag       =  dat[1].data['ip_MAG_APER3'] 
zmag       =  dat[1].data['zp_MAG_APER3']

I427       =  dat[1].data['IB427_MAG_APER3']
I464       =  dat[1].data['IB464_MAG_APER3']
I505       =  dat[1].data['IB505_MAG_APER3']
I574       =  dat[1].data['IB574_MAG_APER3']

##  gmag   =  Vmag + 0.74 * (Bmag - Vmag) - 0.07 (SDSS stellar transform, classic.sdss.org/dr4/algorithms/sdssUBVRITransform.html)
##  gmag   =  -0.025551 * I427 + 0.094313 * I464 + 0.518425 * I505 + 0.414462 * I574 + 0.645
gmag       =  0.193840 * Bmag + 0.813069 * Vmag + 0.302874 ##  (0.106546)

Jmag       =  dat[1].data['J_MAG_APER3']
Hmag       =  dat[1].data['H_MAG_APER3']
Ymag       =  dat[1].data['Y_MAG_APER3']
Kmag       =  dat[1].data['Ksw_MAG_APER3']

##  Klim   =  Kmag < 24.7

##                                                                                                                                                                                                                                                                                                                          
photozs    =  dat[1].data['PHOTOZ']        ##  Median of the photo-z likelihood.  

##  
good       =  INDEEP.astype(bool)

##  Degraded depths. 
degraded_depths = get_depths()
'''
nrows      = 50000

magdict    = dict(zip(['u', 'g', 'r', 'i', 'z'], [umag[good][:nrows], gmag[good][:nrows], rmag[good][:nrows], imag[good][:nrows], zmag[good][:nrows]]))
photozs    = photozs[good][:nrows]
'''
magdict    = dict(zip(['u', 'g', 'r', 'i', 'z'], [umag[good], gmag[good], rmag[good], imag[good], zmag[good]]))
photozs    = photozs[good]

for band in magdict:
    Flux          =  10. ** (-(magdict[band] + 48.60) / 2.5)
    SigF          =  ferr(magdict[band], degraded_depths[band], estar=0.2, alphab=-0.25, alphaf=0.22, lim_snr=None)
    
    mSigF         =  np.ma.masked_invalid(SigF, copy=True)     
    Noise         =  np.random.normal(loc=np.zeros_like(mSigF), scale=mSigF)

    ##  See also eqn. (10.2) of Chromey, Introduction to Observational Astronomy.                                                                                                                                                                                                                                          
    ##  dmag      =  -2.5 * np.log10(Flux + Noise) - 48.60
    lup           =  luptitude(Flux + Noise, SigF)
        
    ##  print('%s \t %.6lf \t %.6lf' % (band, magdict[band], lup))

    if DODEPTH == 'Degraded':
      magdict[band] =  lup

## 
detected  = 0
udetected = 0

## 
if ttype == 'u':
    bcol = 'u-g'
    rcol = 'g-r'

    bmin = -0.5
    bmax =  2.5

    rmin = -0.3
    rmax =  1.2

elif ttype == 'g':
    bcol = 'g-r'
    rcol = 'r-i'

    bmin = -0.5
    bmax =  2.5

    rmin = -0.3
    rmax =  1.2

else:
    raise UserWarning()

##  dN/dz binning. 
zbins     = np.arange(0.0, 6.0, 0.2)

dz        = zbins[1] - zbins[0]
midz      = zbins[:-1] + dz/2.

drop_zs   = []

for i, row in enumerate(magdict['u']):
   mags   = dict(zip(magdict.keys(), [magdict[x][i] for x in magdict.keys()]))
   colors = get_colors(mags, get_colors=['g-r', 'r-i', 'u-g', 'g-r', 'u-z', 'g-i'], fname = None)

   if ttype     == 'u':
       dband        =  mags['r']
       is_detected  = (mags['r'] < degraded_depths['r'])

   elif ttype == 'g':
       dband        =  mags['i']
       is_detected  = (mags['i'] < degraded_depths['i'])

   else:
       is_detected  = False

   if DODEPTH == 'Full':
       is_detected  =  True

   if is_detected:
       detected  += 1
       is_lbg     = colourcut(mags,  dropband=ttype, good=True, fourthlimit=False, BzK_type='all')

       '''
       print('%+.4lf \t %+.4lf \t %+.3lf \t %+.3lf \t %s \t %+.3lf \t %+.3lf' % (mags['u'], mags['g'], mags['r'],\
                                                                                 photozs[i], str(is_lbg), degraded_depths['r'],\
                                                                                 degraded_depths['i']))
       '''

       if is_lbg:
           if plotit:
             cax = plt.scatter(colors[rcol], colors[bcol], c=photozs[i], s=10, vmin=0.0, vmax=5.0, rasterized=True, alpha=0.8)

           if np.isfinite(photozs[i]):  ## (dband > 23.0) 
             drop_zs.append(photozs[i])

       else:
           if plotit:
           ##  if (photozs[i] > 3.5) & (photozs[i] < 4.5) & plotit:  
               draw = np.random.uniform(low=0.0, high=1.0, size=1)
               
               if draw <= 0.01:
                   ##  1% of non-dropouts. 
                   plt.scatter(colors[rcol], colors[bcol], c=photozs[i], marker='x', alpha=0.1, s=20, vmin=0.0, vmax=5.0, rasterized=True)

   else:
       udetected += 1

print('Detection rates: %d \t %d' % (detected, udetected))

##  Histogram of color cut redshifts.                                                                                                             
(dNdz, bins) = np.histogram(drop_zs, bins = zbins)
dNdz         = dNdz.astype(np.float)

output       = np.c_[midz, dNdz]
np.savetxt('dNdz/%s_%sdrops_dz_%.2lf.txt' % (DODEPTH, ttype, dz), output, fmt='%.6le')

##  
if plotit:
  ax     = pl.gca()
  fig    = pl.gcf()

  plt.colorbar(cax, label=r'redshift')

  pl.xlabel(r'$%s$' % rcol)
  pl.ylabel(r'$%s$' % bcol)

  pl.xlim(rmin, rmax)
  pl.ylim(bmin, bmax)

  pl.show()

