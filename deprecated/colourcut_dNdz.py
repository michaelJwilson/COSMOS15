import  os
import  copy
import  glob
import  numpy  as  np
import  pylab  as  pl

from    astropy.table       import Table
from    threedhst.app_mags  import get_colors
from    depths              import get_depths


degraded_depths = get_depths()

def colourcut(mags, dropband='g', good=True, fourthlimit=True, BzK_type='all'):
    if dropband == 'g':
        colors   = get_colors(mags, get_colors=['g-r', 'r-i'], fname = None)
        goodmags = np.all(np.isfinite(np.array([mags['g'], mags['r'], mags['i']])))    ##  Finite magnitudes in relevant bands.                             
                                                            
        return (colors['g-r'] > 1.0) & (colors['r-i'] < 1.0) & (colors['g-r'] > 1.5 * colors['r-i'] + 0.8) & goodmags

    elif dropband == 'u':
        ##  CARS z=3 dropouts.                                                                                                                      
        colors     = get_colors(mags, get_colors=['u-g', 'g-r'], fname = None)
        goodmags   = np.all(np.isfinite(np.array([mags['u'], mags['g'], mags['r']])))  ##  Finite magnitudes in relevant bands.                             
                                                           
        if fourthlimit:
          ##  Section 3 of https://arxiv.org/pdf/0903.3951.pdf
          ##  Note:  Lower limit on (g-r) is a depth dependent statement.  
          return (colors['u-g'] > 1.5) & (colors['g-r'] > -1.0) & (colors['g-r'] < 1.2) & (colors['u-g'] >  1.5 * colors['g-r'] + 0.75) & goodmags

        else:
          ##  Drop Lower limit on (g-r) is a depth dependent statement.
          return (colors['u-g'] > 1.5) & (colors['g-r'] < 1.2) & (colors['u-g'] >  1.5 * colors['g-r'] + 0.75) & goodmags  

    elif dropband == 'BzK':
        ##  Daddi BzK selection, https://arxiv.org/pdf/astro-ph/0409041.pdf.
        colors     = get_colors(mags, get_colors=['z-K', 'B-z'], fname = None)
        goodmags   = np.all(np.isfinite(np.array([mags['z'], mags['K'], mags['B']])))  ##  Finite magnitudes in relevant bands.

        BzK        = colors['z-K'] - colors['B-z']

        if BzK_type   == 'star':
          ##  Actively star-forming galaxies at z > 1.4  
          return (BzK > -0.2) & goodmags

        elif BzK_type == 'passive':
          ##  Passively evolving galaxies at z > 1.4   
          return (BzK < -0.2) & (colors['z-K'] > 2.5) & goodmags

        else:
          return ((BzK > -0.2) & goodmags) | ((BzK < -0.2) & (colors['z-K'] > 2.5) & goodmags)

    elif dropband == 'Euclid':
        ##  u/g dropout selection.  All galaxies returned as true.  
        return  colourcut(mags, dropband='g', good=True) |  colourcut(mags, dropband='u', good=True) 

    elif dropband == 'Synergy':
        ##  No selection.  All galaxies returned as true.                                                                                   
        colors   = get_colors(mags, get_colors=['r-i', 'i-H'], fname = None)

        return  np.ones_like(colors['r-i'], dtype=bool)

    else:
        raise  ValueError('Requested dropband is not available')

def run(DROPTYPE='u', DODEPTH='Full', plotit=True, printit=True, HEAVY='scratch'):
    files         =  glob.glob('colors/%s/*.fits' % HEAVY)
    files         = [x.split('.fits')[0] for x in files]

    files         = [x.split('colors/%s/3DHST_' % HEAVY)[1] for x in files]

    fields        = [x.split('_')[0] for x in files]
    ufields       = list(set(fields))
    nfields       = len(ufields)

    zbins         = np.arange(0.0, 6.0, 0.10)

    dz            = zbins[1] - zbins[0]
    midz          = zbins[:-1] + dz/2.
                                    
    depths        = [xx.split('_')[1].split('_')[0] for xx in files]
    depths        = ['Full' if depth != 'Degraded' else depth for depth in depths]
    udepths       =  set(depths)
    
    results       = {x: np.zeros_like(midz) for x in ufields}
    
    print(DROPTYPE, DODEPTH)
        
    pl.clf()

    for ii, xx in enumerate(files):
        depth  = depths[ii]
        field  = fields[ii]
 
        drops  = []
        
        if (field == 'UVUDF') & (depth == DODEPTH):
            print('Solving for %s (%s    %s).' % (xx, DROPTYPE, DODEPTH))

            magt   = Table.read('colors/%s/3DHST_' % HEAVY + xx + '.fits', format='fits')
            cols   = magt.colnames

            ##  Remove rows with no best-fitting photometric redshift. 
            magt.remove_rows(magt['zpeak'] < -98.)

            mags   = copy.copy(cols)
            mags.remove('id')
            mags.remove('zpeak')
            
            for row in magt:
              if (row['zpeak'] > -99.):
                rowmags = [row[x] for x in mags]  
                magdict = dict(zip(mags, rowmags))
      
                colors  = get_colors(magdict, get_colors=['g-r', 'r-i', 'u-g', 'g-r', 'z-K', 'u-z', 'B-z', 'J-H', 'Y-J', 'g-i', 'i-H'], fname = None)

                if DROPTYPE == 'u':
                  is_detected = (magdict['r'] < degraded_depths['r'])

                elif DROPTYPE == 'g':
                  is_detected = (magdict['i'] < degraded_depths['i'])

                elif DROPTYPE == 'BzK':
                  is_detected = (magdict['K'] < degraded_depths['K'])

                elif DROPTYPE in ['Euclid', 'Synergy']:
                  is_detected = (magdict['H'] < degraded_depths['H'])

                else:
                  is_detected = False

                if DODEPTH == 'Full':
                  is_detected =  True

                
                if is_detected & colourcut(magdict, dropband=DROPTYPE, good=True, fourthlimit=False):
                    drops.append('Y')

                else:
                    drops.append('N')  

            magt['%s-drop' % DROPTYPE]  =  drops
            zs                          =  magt['zpeak'][magt['%s-drop' % DROPTYPE] == 'Y']
      
            ##  Histogram of color cut redshifts. 
            (dNdz, bins)    = np.histogram(zs, bins = zbins)

            dNdz            = dNdz.astype(np.float)
            results[field] += dNdz

            if printit:
                print(magt)

    for field in ['UVUDF']:
        output = np.c_[midz, results[field]]
        np.savetxt('dNdz/%s_%s_%sdrops_dz_%.2lf.txt' % (field, DODEPTH, DROPTYPE, dz), output, fmt='%.6le')
        
        pl.plot(midz, results[field], label=field)
        
    pl.xlabel(r'$z$')
    pl.ylabel(r'$dN/dz$')
    pl.title(r'%s, %s' % (DROPTYPE, DODEPTH))
    pl.legend()
    pl.show(block=True)
        

if __name__ =='__main__':
    print('\n\nWelcome.\n\n')

    plotit        =   True
    printit       =   True
    DODEPTH       =  'Full'     ##  ['Full', 'Degraded']                                                                        
    DROPTYPE      =  'g'        ## ['u', 'g', 'BzK', 'Euclid']                                                                 
    HEAVY         =  'scratch'  ## {'lite', 'scratch', 'test'} 

    for DODEPTH in ['Full']:
      for DROPTYPE in ['u', 'g', 'BzK']:
        run(DROPTYPE=DROPTYPE, DODEPTH=DODEPTH, plotit=True, printit=False, HEAVY=HEAVY)

    print('\n\nDone.\n\n')
