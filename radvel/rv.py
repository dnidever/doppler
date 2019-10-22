#!/usr/bin/env python

"""RV.PY - Generic Radial Velocity Software

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@noao.edu>'
__version__ = '20180922'  # yyyymmdd                                                                                                                           

import os
import numpy as np
import warnings
from astropy.io import fits
from astropy.table import Table, Column
from astropy import modeling
from glob import glob
from scipy.signal import medfilt
from scipy.ndimage.filters import median_filter,gaussian_filter1d,convolve
from scipy.optimize import curve_fit, least_squares
from scipy.special import erf
from scipy.interpolate import interp1d
#from numpy.polynomial import polynomial as poly
#from lmfit import Model
#from apogee.utils import yanny, apload
#from sdss_access.path import path
import thecannon as tc
#import bindata
#from utils import *
from dlnpyutils import utils as dln, bindata

# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

def xcorr_dtype(nlag):
    """Return the dtype for the xcorr structure"""
    dtype = np.dtype([("xshift0",float),("ccp0",float),("xshift",float),("xshifterr",float),
                      ("xshift_interp",float),("ccf",(float,nlag)),("ccferr",(float,nlag)),("ccnlag",int),
                      ("cclag",(int,nlag)),("ccpeak",float),("ccpfwhm",float),("ccp_pars",(float,4)),
                      ("ccp_perror",(float,4)),("ccp_polycoef",(float,4)),("vrel",float),
                      ("vrelerr",float),("w0",float),("dw",float),("chisq",float)])
    return dtype

# astropy.modeling can handle errors and constraints


# Load the cannon model
def load_all_cannon_models():
    fil = os.path.abspath(__file__)
    codedir = os.path.dirname(fil)
    datadir = os.path.dirname(codedir)+'/data/'
    files = glob(datadir+'cannongrid*.pkl')
    nfiles = len(files)
    if nfiles==0:
        raise Exception("No Cannon model files in "+datadir)
    models = []
    for f in files:
        model1 = tc.CannonModel.read(f)
        ranges = np.zeros([3,2])
        for i in range(3):
            ranges[i,:] = minmax(model1._training_set_labels[:,i])
        model1.ranges = ranges
        models.append(model1)
    return models

# Get correct cannon model for a given set of inputs
def get_best_cannon_model(models,pars):
    n = len(models)
    for m in models:
        #inside.append(m.in_convex_hull(pars))        
        # in_convex_hull() is very slow
        inside = True
        for i in range(3):
            inside &= (pars[i]>=m.ranges[i,0]) & (pars[i]<=m.ranges[i,1])
        if inside:
            return m
    
    return None

def model_spectrum(models,teff=None,logg=None,feh=None,wave=None,w0=None,w1=None,dw=None,fwhm=None,trim=False):
    if teff is None:
        raise Exception("Need to input TEFF")    
    if logg is None:
        raise Exception("Need to input LOGG")
    if feh is None:
        raise Exception("Need to input FEH")    
    
    pars = np.array([teff,logg,feh])
    # Get best cannon model
    model = get_best_cannon_model(models,pars)

    npix = model.dispersion.shape[0]
    dw0 = model.dispersion[1]-model.dispersion[0]

    # Wave array input
    if wave is not None:
        w0 = np.min(wave)
        w1 = np.max(wave)
        dw = np.min(slope(wave))
    
    # Defaults for w0, w1, dw
    if w0 is None:
        w0 = np.min(model.dispersion)
    if w1 is None:
        w1 = np.max(model.dispersion)
    if dw is None:
        dw = dw0

    # Trim the model
    if trim is True:
        # Get trimmed Cannon model
        model2 = trim_cannon_model(model,w0=w0-10*dw,w1=w1+10*dw)
        npix2 = model2.dispersion.shape[0]
        # Get the spectrum
        wave1 = model2.dispersion
        flux1 = model2(pars)
    else:
        wave1 = model.dispersion
        flux1 = model(pars)
        npix2 = npix
        
    # Rebin
    if dw != dw0:
        binsize = int(dw // dw0)
        newnpix = int(npix2 // binsize)
        # Need to fix this to include w0 and w1 if possible
        wave2 = rebin(wave1[0:newnpix*binsize],newnpix)
        flux2 = rebin(flux1[0:newnpix*binsize],newnpix)
    else:
        wave2 = wave1
        flux2 = flux1

    # Smoothing
    if fwhm is not None:
        fwhmpix = fwhm/dw
        flux2 = gsmooth(flux2,fwhmpix)
    # Interpolation
    if wave is not None:
        tflux2 = flux2.copy()
        f = interp1d(wave2,tflux2,kind='cubic',bounds_error=False,fill_value=(0.0,0.0),assume_sorted=True)
        flux2 = f(wave)
        del(tflux2)
        wave2 = wave.copy()

    # Create Spec1D object
    spec = Spec1D(flux2,wave=wave2,instrument='Model')
    spec.teff = teff
    spec.logg = logg
    spec.feh = feh
    if fwhm is not None:
        spec.fwhm = fwhm
    spec.snr = np.inf
        
    return spec
    
    
# Object for representing 1D spectra
class Spec1D:
    # Initialize the object
    def __init__(self,flux,err=None,wave=None,mask=None,instrument=None,filename=None,snr=None):
        self.flux = flux
        self.err = None
        self.wave = None
        self.mask = None
        self.instrument = None
        self.filename = None
        self.snr = None
        if err is not None:
            self.err = err
        if wave is not None:
            self.wave = wave
        if mask is not None:
            self.mask = mask
        if instrument is not None:
            self.instrument = instrument
        if filename is not None:
            self.filename = filename
        if snr is not None:
            self.snr = snr
        return

    def __repr__(self):
        s = repr(self.__class__)+"\n"
        if self.instrument is not None:
            s += self.instrument+" spectrum\n"
        if self.filename is not None:
            s += "File = "+self.filename+"\n"
        if self.snr is not None:
            s += ("S/N = %7.2f" % self.snr)+"\n"
        s += "Flux = "+str(self.flux)+"\n"
        if self.err is not None:
            s += "Err = "+str(self.err)+"\n"
        if self.wave is not None:
            s += "Wave = "+str(self.wave)
        return s

    def normalize(self,ncorder=6,perclevel=0.95):
        self._flux = self.flux  # Save the original
        nspec, cont, masked = normspec(self,ncorder=ncorder,perclevel=perclevel)
        self.flux = nspec
        self.cont = cont
        return

    def rv(self,template):
        """Calculate the RV with respect to a template spectrum"""
        pass
        return
        
    def solve(self):
        """Find the RV and stellar parameters of this spectrum"""
        pass
        return

   # maybe add an interp() method to interpolate the
   # spectrum onto a new wavelength scale, outputs a new object

   # a load() or read() method that you can use to read in a spectrum
   # from a file, basically just calls the rdspec() function.
    
# Load a spectrum
def rdspec(filename=None):
    '''
    This reads in a SDSS-IV MWM training set spectrum and returns an
    object that is guaranteed to have certain information.

    Parameters
    ----------
    filename : str
          The filename of the spectrum.

    Returns
    -------
    spec : Spec1D object
        A Spec1D object that always has FLUX, ERR, WAVE, MASK, FILENAME, SPTYPE, WAVEREGIME and INSTRUMENT.

    Example
    -------

    Load an APOGEE apStar spectrum.

    .. code-block:: python

        spec = rdspec("apStar-r8-2M00050083+6349330.fits")

    '''
    if filename is None:
        print("Please input the filename")
        return

    # Different types of spectra
    # apVisit
    # apStar
    # BOSS spec
    # MaStar spectra
    # lp*fits, synthetic spectra
    base, ext = os.path.splitext(os.path.basename(filename))

    # Check that the files exists
    if os.path.exists(filename) is False:
        print(filename+" NOT FOUND")
        return None
    
    # APOGEE apVisit, visit-level spectrum
    if base.find("apVisit") > -1:
        flux = fits.getdata(filename,1)
        spec = Spec1D(flux)
        spec.filename = filename
        spec.sptype = "apVisit"
        spec.waveregime = "NIR"
        spec.instrument = "APOGEE"        
        spec.head = fits.getheader(filename,0)
        spec.err = fits.getdata(filename,2)
        spec.mask = fits.getdata(filename,3)
        spec.wave = fits.getdata(filename,4)
        spec.sky = fits.getdata(filename,5)
        spec.skyerr = fits.getdata(filename,6)
        spec.telluric = fits.getdata(filename,7)
        spec.telerr = fits.getdata(filename,8)
        spec.wcoef = fits.getdata(filename,9)
        spec.lsf = fits.getdata(filename,10)
        spec.meta = fits.getdata(filename,11)   # catalog of RV and other meta-data
        # Spectrum, error, sky, skyerr are in units of 1e-17
        spec.snr = spec.head["SNR"]
        return spec

    # APOGEE apStar, combined spectrum
    if base.find("apStar") > -1:
        # HISTORY APSTAR:  HDU0 = Header only                                             
        # HISTORY APSTAR:  All image extensions have:                                     
        # HISTORY APSTAR:    row 1: combined spectrum with individual pixel weighting     
        # HISTORY APSTAR:    row 2: combined spectrum with global weighting               
        # HISTORY APSTAR:    row 3-nvisits+2: individual resampled visit spectra          
        # HISTORY APSTAR:   unless nvisits=1, which only have a single row                
        # HISTORY APSTAR:  All spectra shifted to rest (vacuum) wavelength scale          
        # HISTORY APSTAR:  HDU1 - Flux (10^-17 ergs/s/cm^2/Ang)                           
        # HISTORY APSTAR:  HDU2 - Error (10^-17 ergs/s/cm^2/Ang)                          
        # HISTORY APSTAR:  HDU3 - Flag mask:                                              
        # HISTORY APSTAR:    row 1: bitwise OR of all visits                              
        # HISTORY APSTAR:    row 2: bitwise AND of all visits                             
        # HISTORY APSTAR:    row 3-nvisits+2: individual visit masks                      
        # HISTORY APSTAR:  HDU4 - Sky (10^-17 ergs/s/cm^2/Ang)                            
        # HISTORY APSTAR:  HDU5 - Sky Error (10^-17 ergs/s/cm^2/Ang)                      
        # HISTORY APSTAR:  HDU6 - Telluric                                                
        # HISTORY APSTAR:  HDU7 - Telluric Error                                          
        # HISTORY APSTAR:  HDU8 - LSF coefficients                                        
        # HISTORY APSTAR:  HDU9 - RV and CCF structure  
        flux = fits.getdata(filename,1)
        spec = Spec1D(flux)
        spec.filename = filename
        spec.sptype = "apStar"
        spec.waveregime = "NIR"
        spec.instrument = "APOGEE"
        spec.head = fits.getheader(filename,0)
        spec.err = fits.getdata(filename,2)
        spec.mask = fits.getdata(filename,3)
        spec.sky = fits.getdata(filename,4)
        spec.skyerr = fits.getdata(filename,5)
        spec.telluric = fits.getdata(filename,6)
        spec.telerr = fits.getdata(filename,7)
        spec.lsf = fits.getdata(filename,8)
        spec.meta = fits.getdata(filename,9)    # meta-data
        # Spectrum, error, sky, skyerr are in units of 1e-17
        #  these are 2D arrays with [Nvisit+2,Npix]
        #  the first two are combined and the rest are the individual spectra
        head1 = fits.getheader(filename,1)
        w0 = head1["CRVAL1"]
        dw = head1["CDELT1"]
        nw = head1["NAXIS1"]
        spec.wave = 10**(np.arange(nw)*dw+w0)
        spec.snr = spec.head["SNR"]
        return spec

    # BOSS spec
    if (base.find("spec-") > -1) | (base.find("SDSS") > -1):
        # HDU1 - binary table of spectral data
        # HDU2 - table with metadata including S/N
        # HDU3 - table with line measurements
        head = fits.getheader(filename,0)
        tab1 = Table.read(filename,1)
        cat1 = Table.read(filename,2)
        spec = Spec1D(tab1["flux"].data[0])
        spec.filename = filename
        spec.sptype = "spec"
        spec.waveregime = "Optical"
        spec.instrument = "BOSS"
        spec.head = head
        # checking for zeros in IVAR
        ivar = tab1["ivar"].data[0].copy()
        bad = (ivar==0)
        if np.sum(bad) > 0:
            ivar[bad] = 1.0
            err = 1.0/np.sqrt(ivar)
            err[bad] = np.nan
        else:
            err = 1.0/np.sqrt(ivar)
        spec.err = err
        spec.ivar = tab1["ivar"].data[0]
        spec.wave = 10**tab1["loglam"].data[0]
        spec.mask = tab1["or_mask"].data[0]
        spec.and_mask = tab1["and_mask"].data[0]
        spec.or_mask = tab1["or_mask"].data[0]
        spec.sky = tab1["sky"].data[0]
        spec.wdisp = tab1["wdisp"].data[0]
        spec.model = tab1["model"].data[0]
        spec.meta = cat1
        # What are the units?
        spec.snr = cat1["SN_MEDIAN_ALL"].data
        return spec        


    # MaStar spec
    if (base.find("mastar-") > -1):
        # HDU1 - table with spectrum and metadata
        tab = Table.read(filename,1)
        spec = Spec1D(tab["FLUX"].data[0])
        spec.filename = filename
        spec.sptype = "MaStar"
        spec.waveregime = "Optical"
        spec.instrument = "BOSS"
        # checking for zeros in IVAR
        ivar = tab["IVAR"].data[0].copy()
        bad = (ivar==0)
        if np.sum(bad) > 0:
            ivar[bad] = 1.0
            err = 1.0/np.sqrt(ivar)
            err[bad] = np.nan
        else:
            err = 1.0/np.sqrt(ivar)
        spec.err = err
        spec.ivar = tab["IVAR"].data[0]
        spec.wave = tab["WAVE"].data[0]
        spec.mask = tab["MASK"].data[0]
        spec.disp = tab["DISP"].data[0]
        spec.presdisp = tab["PREDISP"].data[0]
        meta = {'DRPVER':tab["DRPVER"].data,'MPROCVER':tab["MPROCVER"].data,'MANGAID':tab["MANGAID"].data,'PLATE':tab["PLATE"].data,
                'IFUDESIGN':tab["IFUDESIGN"].data,'MJD':tab["MJD"].data,'IFURA':tab["IFURA"].data,'IFUDEC':tab["IFUDEC"].data,'OBJRA':tab["OBJRA"].data,
                'OBJDEC':tab["OBJDEC"].data,'PSFMAG':tab["PSFMAG"].data,'MNGTARG2':tab["MNGTARG2"].data,'NEXP':tab["NEXP"].data,'HELIOV':tab["HELIOV"].data,
                'VERR':tab["VERR"].data,'V_ERRCODE':tab["V_ERRCODE"].data,'MJDQUAL':tab["MJDQUAL"].data,'SNR':tab["SNR"].data,'PARS':tab["PARS"].data,
                'PARERR':tab["PARERR"].data}
        spec.meta = meta
        # What are the units?
        spec.snr = tab["SNR"].data
        return spec        

    # lp*fits, synthetic spectra
    if base.find("lp") > -1:
        tab = Table.read(filename,format='fits')
        spec = Spec1D(tab["FLUX"].data)
        spec.filename = filename
        spec.sptype = "synthetic"
        spec.wave = tab["WAVE"].data
        spec.mask = np.zeros(len(spec.wave))
        if np.min(spec.wave) < 1e4:
            spec.waveregime = "Optical"
        else:
            spec.waveregime = "NIR"
        spec.instrument = "synthetic"
        # Parse the filename
        # lp0000_XXXXX_YYYY_MMMM_NNNN_Vsini_ZZZZ_SNRKKKK.asc where
        # XXXX gives Teff in K, YYYY is logg in dex (i4.4 format,
        # YYYY*0.01 will give logg in dex),
        # MMMM stands for microturbulent velocity (i4.4 format, fixed at 2.0 km/s),
        # NNNN is macroturbulent velocity (i4.4 format, fixed at 0 km/s),
        # ZZZZ gives projected rotational velocity in km/s (i4.4 format),
        # KKKK refers to SNR (i4.4 format). lp0000 stands for the solar metallicity.
        dum = base.split("_")
        spec.meta = {'Teff':np.float(dum[1]), 'logg':np.float(dum[2])*0.01, 'micro':np.float(dum[3]),
                     'macro':np.float(dum[4]), 'vsini':np.float(dum[6]), 'SNR':np.float(dum[7][3:])}
        spec.snr = spec.meta['SNR']
        spec.err = spec.flux*0.0+1.0/spec.snr
        return spec      
    

def ccorrelate(x, y, lag, yerr=None, covariance=False, double=None, nomean=False):
    """This function computes the cross correlation of two samples.

    This function computes the cross correlation Pxy(L) or cross
    covariance Rxy(L) of two sample populations X and Y as a function
    of the lag (L).

    This was translated from APC_CORRELATE.PRO which was itself a
    modification to the IDL C_CORRELATE.PRO function.

    Parameters
    ----------
    x : array
      The first array to cross correlate.
    y : array
      The second array to cross correlate.  Must be same length as x.
    lag : array
      Vector that specifies the absolute distance(s) between
             indexed elements of X in the interval [-(n-2), (n-2)].
    yerr : array, optional
       Array of uncertainties in Y
    covariange : bool
        If true, then the sample cross covariance is computed.

    Returns
    -------
    cross : array
         The cross correlation or cross covariance.
    cerror : array
         The uncertainty in "cross".  Only if "yerr" is input.

    Example
    -------

    Define two n-element sample populations.

    .. code-block:: python

         x = [3.73, 3.67, 3.77, 3.83, 4.67, 5.87, 6.70, 6.97, 6.40, 5.57]
         y = [2.31, 2.76, 3.02, 3.13, 3.72, 3.88, 3.97, 4.39, 4.34, 3.95]

    Compute the cross correlation of X and Y for LAG = -5, 0, 1, 5, 6, 7

    .. code-block:: python

         lag = [-5, 0, 1, 5, 6, 7]
         result = ccorrelate(x, y, lag)

    The result should be:

    .. code-block:: python

         [-0.428246, 0.914755, 0.674547, -0.405140, -0.403100, -0.339685]

    """


    # Compute the sample cross correlation or cross covariance of
    # (Xt, Xt+l) and (Yt, Yt+l) as a function of the lag (l).

    nx = len(x)

    if (nx != len(y)):
        raise ValueError("X and Y arrays must have the same number of elements.")

    # Check length.
    if (nx<2):
        raise ValueError("X and Y arrays must contain 2 or more elements.")

    # Remove the mean
    if nomean is False:
        xmn = np.nanmean(x)
        ymn = np.nanmean(y)
        xd = x.copy()-xmn
        yd = y.copy()-ymn
    else:
        xd = x.copy()
        yd = y.copy()
    if yerr is not None: yerr2=yerr.copy()
        
    fx = np.isfinite(x)
    ngdx = np.sum(fx)
    nbdx = np.sum((fx==False))
    if nbdx>0: xd[(fx==False)]=0.0
    fy = np.isfinite(y)
    ngdy = np.sum(fy)
    nbdy = np.sum((fy==False))
    if nbdy>0:
        yd[(fy==False)]=0.0
        if yerr is not None: yerr2[(fy==False)]=0.0
    nlag = len(lag)

    cross = np.zeros(nlag,dtype=float)
    cross_error = np.zeros(nlag,dtype=float)
    num = np.zeros(nlag,dtype=int)  # number of "good" points at this lag
    for k in range(nlag):
        # Note the reversal of the variables for negative lags.
        if lag[k]>0:
             cross[k] = np.sum(xd[0:nx - lag[k]] * yd[lag[k]:])
             num[k] = np.sum(fx[0:nx - lag[k]] * fy[lag[k]:]) 
             if yerr is not None:
                 cross_error[k] = np.sum( (xd[0:nx - lag[k]] * yerr2[lag[k]:])**2 )
        else:
             cross[k] =  np.sum(yd[0:nx + lag[k]] * xd[-lag[k]:])
             num[k] = np.sum(fy[0:nx + lag[k]] * fx[-lag[k]:])
             if yerr is not None:
                 cross_error[k] = np.sum( (yerr2[0:nx + lag[k]] * xd[-lag[k]:])**2 )
    # Normalize by number of "good" points
    cross *= np.max(num)
    pnum = (num>0)
    cross[pnum] /= num[pnum]  # normalize by number of "good" points
    # Take sqrt to finish adding errors in quadrature
    cross_error = np.sqrt(cross_error)
    # normalize
    cross_error *= np.max(num)
    cross_error[pnum] /= num[pnum]
    
    # Divide by N for covariance, or divide by variance for correlation.
    if ngdx>2:
        rmsx = np.sqrt(np.sum(xd[fx]**2))
    else:
        rmsx=1.0
    if rmsx==0.0: rmsx=1.0
    if ngdy>2:
        rmsy = np.sqrt(np.sum(yd[fy]**2))
    else:
        rmsy=1.0
    if rmsy==0.0: rmsy=1.0
    if covariance is True:
        cross /= nx
        cross_error /= nx
    else:
        cross /= rmsx*rmsy
        cross_error /= rmsx*rmsy

    if yerr is not None: return cross, cross_error
    return cross

def robust_correlate(x, y, lag, yerr=None, covariance=False, double=None, nomean=False):
    """This function computes the cross correlation of two samples.

    This function computes the cross correlation Pxy(L) or cross
    covariance Rxy(L) of two sample populations X and Y as a function
    of the lag (L).

    This was translated from APC_CORRELATE.PRO which was itself a
    modification to the IDL C_CORRELATE.PRO function.

    Parameters
    ----------
    x : array
      The first array to cross correlate.
    y : array
      The second array to cross correlate.  Must be same length as x.
    lag : array
      Vector that specifies the absolute distance(s) between
             indexed elements of X in the interval [-(n-2), (n-2)].
    yerr : array, optional
       Array of uncertainties in Y
    covariange : bool
        If true, then the sample cross covariance is computed.

    Returns
    -------
    cross : array
         The cross correlation or cross covariance.
    cerror : array
         The uncertainty in "cross".  Only if "yerr" is input.

    Example
    -------

    Define two n-element sample populations.

    .. code-block:: python

         x = [3.73, 3.67, 3.77, 3.83, 4.67, 5.87, 6.70, 6.97, 6.40, 5.57]
         y = [2.31, 2.76, 3.02, 3.13, 3.72, 3.88, 3.97, 4.39, 4.34, 3.95]

    Compute the cross correlation of X and Y for LAG = -5, 0, 1, 5, 6, 7

    .. code-block:: python

         lag = [-5, 0, 1, 5, 6, 7]
         result = ccorrelate(x, y, lag)

    The result should be:

    .. code-block:: python

         [-0.428246, 0.914755, 0.674547, -0.405140, -0.403100, -0.339685]

    """


    # Compute the sample cross correlation or cross covariance of
    # (Xt, Xt+l) and (Yt, Yt+l) as a function of the lag (l).

    nx = len(x)

    if (nx != len(y)):
        raise ValueError("X and Y arrays must have the same number of elements.")

    # Check length.
    if (nx<2):
        raise ValueError("X and Y arrays must contain 2 or more elements.")

    # Remove the mean
    if nomean is False:
        xmn = np.nanmean(x)
        ymn = np.nanmean(y)
        xd = x.copy()-xmn
        yd = y.copy()-ymn
    else:
        xd = x.copy()
        yd = y.copy()
    yerr2 = np.ones(x.shape)
    if yerr is not None: yerr2=yerr.copy()
    
        
    fx = np.isfinite(x)
    ngdx = np.sum(fx)
    nbdx = np.sum((fx==False))
    if nbdx>0: xd[(fx==False)]=0.0
    fy = np.isfinite(y)
    ngdy = np.sum(fy)
    nbdy = np.sum((fy==False))
    if nbdy>0:
        yd[(fy==False)]=0.0
        yerr2[(fy==False)]=np.inf
    nlag = len(lag)

    loss = np.zeros(nlag,dtype=float)+1e30  # all bad to start
    #loss_error = np.zeros(nlag,dtype=float)
    num = np.zeros(nlag,dtype=int)  # number of "good" points at this lag
    for k in range(nlag):
        # Note the reversal of the variables for negative lags.
        if lag[k]>0:
             loss[k] = np.sum(np.abs((xd[0:nx - lag[k]] - yd[lag[k]:])/yerr2[lag[k]:]))
             num[k] = np.sum(fx[0:nx - lag[k]] * fy[lag[k]:]) 
             #if yerr is not None:
             #    loss_error[k] = np.sum( (xd[0:nx - lag[k]] * yerr2[lag[k]:])**2 )
        else:
             loss[k] =  np.sum(np.abs((yd[0:nx + lag[k]] - xd[-lag[k]:])/yerr2[0:nx + lag[k]]))
             num[k] = np.sum(fy[0:nx + lag[k]] * fx[-lag[k]:])
             #if yerr is not None:
             #    loss_error[k] = np.sum( (yerr2[0:nx + lag[k]] * xd[-lag[k]:])**2 )
    # Normalize by number of "good" points
    #loss *= np.max(num)
    pnum = (num>0)
    loss[pnum] /= num[pnum]  # normalize by number of "good" points
    ## Take sqrt to finish adding errors in quadrature
    #loss_error = np.sqrt(loss_error)
    ## normalize
    #loss_error *= np.max(num)
    #loss_error[pnum] /= num[pnum]
    
    # Divide by N for covariance, or divide by variance for correlation.
    #if ngdx>2:
    #    rmsx = np.sqrt(np.sum(xd[fx]**2))
    #else:
    #    rmsx=1.0
    #if rmsx==0.0: rmsx=1.0
    #if ngdy>2:
    #    rmsy = np.sqrt(np.sum(yd[fy]**2))
    #else:
    #    rmsy=1.0
    #if rmsy==0.0: rmsy=1.0
    #if covariance is True:
    #    loss /= nx
    #    loss_error /= nx
    #else:
    #    loss /= rmsx*rmsy
    #    loss_error /= rmsx*rmsy

    return loss

def specshift_resid(xshift,spec,temp,sigma=1.0):
    """ This shifts a template spectrum and then returns the
    residual when compared to an observed spectrum"""
    sig = sigma
    if sigma is None: sig=1.0
    x = np.arange(len(spec))
    f = interp1d(x,temp,kind='cubic',bounds_error=False,fill_value=(0.0,0.0),assume_sorted=True)
    shtemp = f(x+xshift)
    resid = (spec-shtemp)/sig
    print(xshift)
    return resid

def robust_xcorr(spec, temp, err=None, maxlag=200):
    """ This program performs robust shifting/cross-correlation
    of two spectra/arrays."""

    bounds = (-maxlag,maxlag)
    res_robust = least_squares(specshift_resid, 0.0, loss='soft_l1', f_scale=0.1, args=(spec,temp,err), bounds=bounds)
    if res_robust.success is False:
        raise Exception("Problem with least squares polynomial fitting. Status="+str(res_robust.status))
    coef = res_robust.x

    return coef

def normspec(spec=None,ncorder=6,fixbadpix=True,noerrcorr=False,
             binsize=0.05,perclevel=95.0,growsky=False,nsky=5):
    """
    NORMSPEC

    This program normalizes a spectrum

    Parameters
    ----------
    spec : Spec1D object
           A spectrum object.  This at least needs
                to have a FLUX and WAVE attribute.
    ncorder : int, default=6
            The continuum polynomial order.  The default is 6.
    noerrcorr : bool, default=False
            Do not use a correction for the effects of the errors
            on the continuum measurement.  The default is to make
            this correction if errors are included.
    fixbadpix : bool, default=True
            Set bad pixels to the continuum
    binsize : float, default=0.05
            The binsize to use (in units of 900A) for determining
            the Nth percentile spectrum to fit with a polynomial.

    perclevel : float, default=95
            The Nth percentile to use to determine the continuum.

    Returns
    -------
    nspec : array
         The continuum normalized spectrum.
    cont : array
         The continuum array.
    masked : array
         A boolean array specifying if a pixel was masked (True) or not (False).

    """

    # Not enough inputs
    if spec is None:
        raise ValueError("""spec2 = apnormspec(spec,fixbadpix=fixbadpix,ncorder=ncorder,noerrcorr=noerrcorr,
                                             binsize=binsize,perclevel=perclevel)""")
    musthave = ['flux','err','mask','wave']
    for a in musthave:
        if hasattr(spec,a) is False:
            raise ValueError("spec object must have "+a)

    # Can only 1D or 2D arrays
    if spec.flux.ndim>2:
        raise Exception("Flux can only be 1D or 2D arrays")
        
    # Do special processing if the input is 2D
    #  Loop over the shorter axis
    if spec.flux.ndim==2:
        nx, ny = spec.flux.shape
        nspec = np.zeros(spec.flux.shape)
        cont = np.zeros(spec.flux.shape)
        masked = np.zeros(spec.flux.shape,bool)
        if nx<ny:
            for i in range(nx):
                flux = spec.flux[i,:]
                err = spec.err[i,:]
                mask = spec.mask[i,:]
                wave = spec.wave[i,:]
                spec1 = Spec1D(flux)
                spec1.err = err
                spec1.mask = mask
                spec1.wave = wave
                nspec1, cont1, masked1 = normspec(spec1,fixbadpix=fixbadpix,ncorder=ncorder,noerrcorr=noerrcorr,
                                                  binsize=binsize,perclevel=perclevel)
                nspec[i,:] = nspec1
                cont[i,:] = cont1
                masked[i,:] = masked1                
        else:
            for i in range(ny):
                flux = spec.flux[:,i]
                err = spec.err[:,i]
                mask = spec.mask[:,i]
                wave = spec.wave[:,i]
                spec1 = Spec1D(flux)
                spec1.err = err
                spec1.mask = mask
                spec1.wave = wave
                nspec1, cont1, masked1 = normspec(spec1,fixbadpix=fixbadpix,ncorder=ncorder,noerrcorr=noerrcorr,
                                                  binsize=binsize,perclevel=perclevel)
                nspec[:,i] = nspec1
                cont[:,i] = cont1
                masked[:,i] = masked1                
        return (nspec,cont,masked)
                
        
    # Continuum Normalize
    #----------------------
    w = spec.wave.copy()
    x = (w-np.median(w))/(np.max(w*0.5)-np.min(w*0.5))  # -1 to +1
    y = spec.flux.copy()
    yerr = None
    if hasattr(spec,'err') is True:
        if spec.err is not None:
            yerr = spec.err.copy()
            
    # Get good pixels, and set bad pixels to NAN
    #--------------------------------------------
    gdmask = (y>0)        # need positive fluxes
    ytemp = y.copy()

    # Exclude pixels with mask=bad
    if hasattr(spec,'mask') is True:
        if spec.mask is not None:
            mask = spec.mask.copy()
            gdmask = (mask == 0)
    gdpix = (gdmask == 1)
    ngdpix = np.sum(gdpix)
    bdpix = (gdmask != 1)
    nbdpix = np.sum(bdpix)
    if nbdpix>0: ytemp[bdpix]=np.nan   # set bad pixels to NAN for now

    # First attempt at continuum
    #----------------------------
    # Bin the data points
    xr = [np.nanmin(x),np.nanmax(x)]
    bins = np.ceil((xr[1]-xr[0])/binsize)+1
    ybin, bin_edges, binnumber = bindata.binned_statistic(x,ytemp,statistic='percentile',
                                                          percentile=perclevel,bins=bins,range=None)
    xbin = bin_edges[0:-1]+0.5*binsize
    gdbin = np.isfinite(ybin)
    ngdbin = np.sum(gdbin)
    if ngdbin<(ncorder+1):
        raise Exception("Not enough good flux points to fit the continuum")
    # Fit with robust polynomial
    coef1 = poly_fit(xbin[gdbin],ybin[gdbin],ncorder,robust=True)
    cont1 = poly(x,coef1)

    # Subtract smoothed error from it to remove the effects
    #  of noise on the continuum measurement
    if (yerr is not None) & (noerrcorr is False):
        smyerr = medfilt(yerr,151)                            # first median filter
        smyerr = gsmooth(smyerr,100)                          # Gaussian smoothing
        coef_err = poly_fit(x,smyerr,ncorder,robust=True)     # fit with robust poly
        #poly_err = poly(x,coef_err)
        #cont1 -= 2*poly_err   # is this right????
        med_yerr = np.median(smyerr)                          # median error
        cont1 -= 2*med_yerr

    # Second iteration
    #-----------------
    #  This helps remove some residual structure
    ytemp2 = ytemp/cont1
    ybin2, bin_edges2, binnumber2 = bindata.binned_statistic(x,ytemp2,statistic='percentile',
                                                             percentile=perclevel,bins=bins,range=None)
    xbin2 = bin_edges2[0:-1]+0.5*binsize
    gdbin2 = np.isfinite(ybin2)
    ngdbin2 = np.sum(gdbin2)
    if ngdbin2<(ncorder+1):
        raise Exception("Not enough good flux points to fit the continuum")
    # Fit with robust polynomial
    coef2 = poly_fit(xbin2[gdbin2],ybin2[gdbin2],ncorder,robust=True)
    cont2 = poly(x,coef2)

    # Subtract smoothed error again
    if (yerr is not None) & (noerrcorr is False):    
      cont2 -= med_yerr/cont1

    # Final continuum
    cont = cont1*cont2  # final continuum

    # "Fix" bad pixels
    if (nbdpix>0) & fixbadpix is True:
        y[bdpix] = cont[bdpix]

    # Create continuum normalized spectrum
    nspec = spec.flux.copy()/cont

    # Add "masked" array
    masked = np.zeros(spec.flux.shape,bool)
    if (fixbadpix is True) & (nbdpix>0):
        masked[bdpix] = True
    
    return (nspec,cont,masked)


def specxcorr(wave=None,tempspec=None,obsspec=None,obserr=None,maxlag=200,errccf=False,prior=None):
    """This measures the radial velocity of a spectrum vs. a template using cross-correlation.

    This program measures the cross-correlation shift between
    a template spectrum (can be synthetic or observed) and
    an observed spectrum (or multiple spectra) on the same
    logarithmic wavelength scale.

    Parameters
    ----------
    wave : array
          The wavelength array.
    tempspec :
          The template spectrum: normalized and on log-lambda scale.
    obsspec : array
           The observed spectra: normalized and sampled on tempspec scale.
    obserr : array
           The observed error; normalized and sampled on tempspec scale.
    maxlag : int
           The maximum lag or shift to explore.
    prior : array, optional 
           Set a Gaussian prior on the cross-correlation.  The first
           term is the central position (in pixel shift) and the
           second term is the Gaussian sigma (in pixels).

    Returns
    -------
    outstr : numpy structured array
           The output structure of the final derived RVs and errors.
    auto : array
           The auto-correlation function of the template

    Examples
    --------

    out = apxcorr(wave,tempspec,spec,err)
    
    """
    
    cspeed = 2.99792458e5  # speed of light in km/s

    # Not enough inputs
    if (wave is None) | (tempspec is None) | (obsspec is None) | (obserr is None):
        raise ValueError('Syntax - out = apxcorr(wave,tempspec,spec,err,auto=auto)')
        return

    nwave = len(wave)
    
    # Are there multiple observed spectra
    if obsspec.ndim>1:
        nspec = obsspec.shape[1]
    else:
        nspec = 1
    
    # Set up the cross-correlation parameters
    #  this only gives +/-450 km/s with 2048 pixels, maybe use larger range
    nlag = 2*np.round(np.abs(maxlag))+1
    if ((nlag % 2) == 0): nlag +=1  # make sure nlag is even
    dlag = 1
    minlag = -nlag/2
    lag = np.arange(nlag)*dlag+minlag

    # Initialize the output structure
    outstr = np.zeros(1,dtype=xcorr_dtype(nlag))
    outstr["xshift"] = np.nan
    outstr["xshifterr"] = np.nan
    outstr["vrel"] = np.nan
    outstr["vrelerr"] = np.nan
    outstr["chisq"] = np.nan

    # Interpolate the visit spectrum onto the synth wavelength grid
    #---------------------------------------------------------------
    wobs = wave.copy()
    nw = len(wobs)
    spec = obsspec.copy()
    err = obserr.copy()
    template = tempspec.copy()

    ## Find first and last non-zero pixels in observed spec
    #gd, = np.where( (spec != 0.0) & (np.isfinite(spec)))
    #ngd = np.sum(gd)
    #if ngd==0:
    #    raise Exception("No good pixels")
    #goodlo = (0 if gd[0]<0 else gd[0])
    #goodhi = ((nw-1) if gd[ngd-1]<(nw-1) else gd[ngd-1])

    # mask bad pixels, set to NAN
    sfix = (spec < 0.01)
    nsfix = np.sum(sfix)
    if nsfix>0: spec[sfix] = np.nan
    tfix = (template < 0.01)
    ntfix = np.sum(tfix)
    if ntfix>0: template[tfix] = np.nan

    # set cross-corrlation window to be good range + nlag
    #lo = (0 if (gd[0]-nlag)<0 else gd[0]-nlag)
    #hi = ((nw-1) if (gd[ngd-1]+nlag)>(nw-1) else gd[ngd-1]+nlag)

    indobs, = np.where(np.isfinite(spec) == True)  # only finite values, in case any NAN
    nindobs = np.sum(indobs)
    indtemp, = np.where(np.isfinite(template) == True)
    nindtemp = np.sum(indtemp)
    if (nindobs>0) & (nindtemp>0):
        # Cross-Correlation
        #------------------
        # Calculate the CCF uncertainties using propagation of errors
        # Make median filtered error array
        #   high error values give crazy values in ccferr
        obserr1 = err.copy()
        bderr = ((obserr1 > 1) | (obserr1 <= 0.0))
        nbderr = np.sum(bderr)
        ngderr = np.sum((bderr==False))
        if (nbderr > 0) & (ngderr > 1): obserr1[bderr]=np.median([obserr1[(bderr==False)]])
        obserr1 = median_filter(obserr1,51)
        ccf, ccferr = ccorrelate(template,obsspec,lag,obserr1)

        # Apply flat-topped Gaussian prior with unit amplitude
        #  add a broader Gaussian underneath so the rest of the
        #   CCF isn't completely lost
        if prior is not None:
            ccf *= np.exp(-0.5*(((lag-prior[0])/prior[1])**4))*0.8+np.exp(-0.5*(((lag-prior[0])/150)**2))*0.2
        
    else:   # no good pixels
        ccf = np.float(lag)*0.0
        if (errccf is True) | (nofit is False): ccferr=ccf

    # Remove the median
    ccf -= np.median(ccf)

    # Best shift
    best_shiftind0 = np.argmax(ccf)
    best_xshift0 = lag[best_shiftind0]
    #temp = shift( tout, best_xshift0)
    temp = np.roll(template, best_xshift0)

    # Find Chisq for each synthetic spectrum
    gdpix, = np.where( (np.isfinite(spec)==True) & (np.isfinite(template)==True) & (spec>0.0) & (err>0.0) & (err < 1e5))
    ngdpix = np.sum(gdpix)
    if (ngdpix==0):
        raise Exception('Bad spectrum')
    chisq = np.sqrt( np.sum( (spec[gdpix]-template[gdpix])**2/err[gdpix]**2 )/ngdpix )

    outstr["chisq"] = chisq
    outstr["ccf"] = ccf
    outstr["ccferr"] = ccferr
    outstr["ccnlag"] = nlag
    outstr["cclag"] = lag    
    
    # Remove smooth background at large scales
    cont = gaussian_filter1d(ccf,100)
    ccf_diff = ccf-cont

    # Get peak of CCF
    best_shiftind = np.argmax(ccf_diff)
    best_xshift = lag[best_shiftind]
    
    # Fit ccf peak with a Gaussian plus a line
    #---------------------------------------------
    # Some CCF peaks are SOOO wide that they span the whole width
    # do the first one without background subtraction
    estimates0 = [ccf_diff[best_shiftind0], best_xshift0, 4.0, 0.0]
    lbounds0 = [1e-3, np.min(lag), 0.1, -np.inf]
    ubounds0 =  [np.inf, np.max(lag), np.max(lag), np.inf]
    pars0, cov0 = gaussfit(lag,ccf_diff,estimates0,ccferr,bounds=(lbounds0,ubounds0))
    perror0 = cov0[np.arange(len(pars0)),np.arange(len(pars0))]

    # Fit the width
    #  keep height, center and constant constrained
    estimates1 = pars0
    estimates1[1] = best_xshift
    lbounds1 = [0.5*estimates1[0], best_xshift-4, 0.3*estimates1[2], lt(np.min(ccf_diff),lt(0,estimates1[3]-0.1)) ]
    ubounds1 =  [1.5*estimates1[0], best_xshift+4, 1.5*estimates1[2], gt(np.max(ccf_diff)*0.5,estimates1[3]+0.1) ]
    lo1 = np.int(gt(np.floor(best_shiftind-gt(estimates1[2]*2,5)),0))
    hi1 = np.int(lt(np.ceil(best_shiftind+gt(estimates1[2]*2,5)),len(lag)))
    pars1, cov1 = gaussfit(lag[lo1:hi1],ccf_diff[lo1:hi1],estimates1,ccferr[lo1:hi1],bounds=(lbounds1,ubounds1))
    yfit1 = gaussian(lag[lo1:hi1],*pars1)
    perror1 = cov1[np.arange(len(pars1)),np.arange(len(pars1))]
    
    # Fefit and let constant vary more, keep width constrained
    estimates2 = pars1
    estimates2[3] = np.median(ccf_diff[lo1:hi1]-yfit1) + pars1[3]
    lbounds2 = [0.5*estimates2[0], limit(best_xshift-gt(estimates2[2],1), np.min(lag), estimates2[1]-1),
                0.3*estimates2[2], lt(np.min(ccf_diff),lt(0,estimates2[3]-0.1)) ]
    ubounds2 = [1.5*estimates2[0], limit(best_xshift+gt(estimates2[2],1), estimates2[1]+1, np.max(lag)),
                1.5*estimates2[2], gt(np.max(ccf_diff)*0.5,estimates2[3]+0.1) ]
    lo2 = np.int(gt(np.floor( best_shiftind-gt(estimates2[2]*2,5)),0))
    hi2 = np.int(lt(np.ceil( best_shiftind+gt(estimates2[2]*2,5)),len(lag)))
    pars2, cov2 = gaussfit(lag[lo2:hi2],ccf_diff[lo2:hi2],estimates2,ccferr[lo2:hi2],bounds=(lbounds2,ubounds2))
    yfit2 = gaussian(lag[lo2:hi2],*pars2)    
    perror2 = cov2[np.arange(len(pars2)),np.arange(len(pars2))]
    
    # Refit with even narrower range
    estimates3 = pars2
    estimates3[3] = np.median(ccf_diff[lo1:hi1]-yfit1) + pars1[3]
    lbounds3 = [0.5*estimates3[0], limit(best_xshift-gt(estimates3[2],1), np.min(lag), estimates3[1]-1),
                0.3*estimates3[2], lt(np.min(ccf_diff),lt(0,estimates3[3]-0.1)) ]
    ubounds3 = [1.5*estimates3[0], limit(best_xshift+gt(estimates3[2],1), estimates3[1]+1, np.max(lag)),
                1.5*estimates3[2], gt(np.max(ccf_diff)*0.5,estimates3[3]+0.1) ]
    lo3 = np.int(gt(np.floor(best_shiftind-gt(estimates3[2]*2,5)),0))
    hi3 = np.int(lt(np.ceil(best_shiftind+gt(estimates3[2]*2,5)),len(lag)))
    pars3, cov3 = gaussfit(lag[lo3:hi3],ccf_diff[lo3:hi3],estimates3,ccferr[lo3:hi3],bounds=(lbounds3,ubounds3))
    yfit3 = gaussian(lag[lo3:hi3],*pars3)    
    perror3 = cov3[np.arange(len(pars3)),np.arange(len(pars3))]

    # This seems to fix high shift/sigma errors
    if (perror3[0]>10) | (perror3[1]>10):
        dlbounds3 = [0.5*estimates3[0], -10+pars3[1], 0.01, lt(np.min(ccf_diff),lt(0,estimates3[3]-0.1)) ]
        dubounds3 = [1.5*estimates3[0], 10+pars3[1], 2*pars3[2], gt(np.max(ccf_diff)*0.5,estimates3[3]+0.1) ]
        dpars3, dcov3 = gaussfit(lag[lo3:hi3],ccf_diff[lo3:hi3],pars3,ccferr[lo3:hi3],bounds=(dlbounds3,dubounds3))
        dyfit3 = gaussian(lag[lo3:hi3],*pars3)    
        perror3 = dcov3[np.arange(len(pars3)),np.arange(len(pars3))]
        
    # Final parameters
    pars = pars3
    perror = perror3
    xshift = pars[1]
    xshifterr = perror[1]
    ccpfwhm_pix = pars[2]*2.35482  # ccp fwhm in pixels
    # v = (10^(delta log(wave))-1)*c
    dwlog = np.median(slope(np.log10(wave)))
    ccpfwhm = ( 10**(ccpfwhm_pix*dwlog)-1 )*cspeed  # in km/s

    # Convert pixel shift to velocity
    #---------------------------------
    # delta log(wave) = log(v/c+1)
    # v = (10^(delta log(wave))-1)*c
    dwlog = np.median(slope(np.log10(wave)))
    vrel = ( 10**(xshift*dwlog)-1 )*cspeed
    # Vrel uncertainty
    dvreldshift = np.log(10.0)*(10**(xshift*dwlog))*dwlog*cspeed  # derivative wrt shift
    vrelerr = dvreldshift * xshifterr

    # Make CCF structure and add to STR
    #------------------------------------
    outstr["xshift0"] = best_xshift
    outstr["ccp0"] = np.max(ccf)
    outstr["xshift"] = xshift
    outstr["xshifterr"] = xshifterr
    #outstr[i].xshift_interp = xshift_interp
    outstr["ccpeak"] = pars[0] 
    outstr["ccpfwhm"] = ccpfwhm  # in km/s
    outstr["ccp_pars"] = pars
    outstr["ccp_perror"] = perror
    #outstr[i].ccp_polycoef = polycoef
    outstr["vrel"] = vrel
    outstr["vrelerr"] = vrelerr
    outstr["w0"] = np.min(wave)
    outstr["dw"] = dwlog
    
    return outstr


def apxcorr(wave,tempspec,obsspec,obserr,outstr,dosum=False,maxlag=None,nofit=True,
            pixlim=None,errccf=False):
    """
    This program measures the cross-correlation shift between
    a template spectrum (can be synthetic or observed) and
    an observed spectrum (or multiple spectra) on the same
    logarithmic wavelength scale.

    Parameters
    ----------
    wave : array
         Wavelength array
    tempspec : array
         Template spectrum: normalized and on log-lambda scale
    obsspec : array
         Observed spectra: normalized and sampled on tempspec scale
    obserr : array
         Observed error; normalized and sampled on tempspec scale
    dosum : bool, optional, default=False
         Set to sum the multiple cross-correlation functions; used
         if we have three chips of data that are stacked

    Returns
    -------
    outstr : numpy structured array
         The output structured array of the final derived RVs and errors.
    auto : array
         The auto-correlation function of the template

    Usage
    -----
    >>>outstr, auto = apxcorr(wave,tempspec,spec,err,outstr)
    
    By D.Nidever  June 2012
    By J.Holtzman  October 2012
    """
    
    cspeed = 2.99792458e5  # speed of light in km/s

    nwave = len(wave)
    ntempspec = len(tempspec)
    nspec = len(obsspec)
    nerr = len(obserr)

    # Not enough inputs
    if (nwave==0) | (ntempspec==0) | (nspec==0) | (nerr==0):
        raise ValueError('Syntax - apxcorr,wave,tempspec,spec,err,outstr,auto=auto,pl=pl')
        return

    sz = size(obsspec)
    if sz[0]==2:
        nspec = sz[2]
    else:
        nspec = 1
    nsum = 1
    if dosum is True:
      nsum = nspec
      nspec = 1
    if pixlim is not None: nsum = len(pixlim[0,:])

    # Set up the cross-correlation parameters
    #  this only gives +/-450 km/s with 2048 pixels, maybe use larger range
    nlag = 401
    if len(maxlag)>0: nlag = 2*np.round(np.abs(maxlag[0]))+1
    if ((nlag % 2) == 0): nlag+=1  # make sure nlag is even
    dlag = 1
    minlag = -nlag/2
    lag = np.arange(nlag)*dlag+minlag


    # Spectrum cross-correlation loop
    #---------------------------------
    for i in range(nspec):

        nsumgood = 0
        for ichip in range(nsum):
            # Interpolate the visit spectrum onto the synth wavelength grid
            #---------------------------------------------------------------
            wobs = reform(wave)
            nw = len(wobs)
            if pixlim is not None:
                if pixlim[0,ichip] < pixlim[1,ichip]:
                    spec = obsspec[pixlim[0,ichip]:pixlim[1,ichip],i]
                    err = obserr[pixlim[0,ichip]:pixlim[1,ichip],i]
                    template = tempspec[pixlim[0,ichip]:pixlim[1,ichip]]
                else:
                    spec = 0.
                    err = 0.
                    template = 0.
                nw = len(spec)
            else:
                spec = obsspec[:,i+ichip]
                err = obserr[:,i+ichip]
                template = tempspec

            # find first and last non-zero pixels in observed spec
            gd, = np.where(spec != 0.0)
            ngd = np.sum(gd)
            if ngd==0:
                ccf = np.float(lag)*0.0
                if (errccf is True) | (nofit is False): ccferr = ccf
                lo = 0
                hi = nw-1
                goto,BOMB1
            goodlo = (0 if gd[0]<0 else gd[0])
            goodhi = ((nw-1) if gd[ngd-1]<(nw-1) else gd[ngd-1])

            # mask bad pixels, set to NAN
            sfix, = np.where(spec < 0.01)
            nsfix = np.sum(sfix)
            if nsfix>0: spec[sfix] = np.nan
            tfix, = np.where(template < 0.01)
            ntfix = np.sum(tfix)
            if ntfix>0: template[tfix] = np.nan

            # How masking pixels affects the cross-correlation
            # -mean
            # -total(x*y)
            # -1/rmsx*rmsy
            # apc_correlate deals with all this appropriately if bad pixels are NAN

            # set cross-corrlation window to be good range + nlag
            lo = (0 if (gd[0]-nlag)<0 else gd[0]-nlag)
            hi = ((nw-1) if (gd[ngd-1]+nlag)>(nw-1) else gd[ngd-1]+nlag)

            indobs, = np.where(np.isfinite(spec) == 1)  # only finite values, in case any NAN
            nindobs = np.sum(indobs)
            indtemp, = np.where(np.isfinite(template) == 1)
            nindtemp = np.sum(indtemp)
            if ((hi-lo+1)>nlag) & (nindobs>0) & (nindtemp>0):
                # Cross-Correlation
                #------------------
                ccf = ccorrelate(template[lo:hi],spec[lo:hi],lag)

                # Calculate the CCF uncertainties using propagation of errors
                #if keyword_set(errccf) then begin
                if (errccf is True) | (nofit is False):
                    # Make median filtered error array
                    #   high error values give crazy values in ccferr
                    obserr1 = err[lo:hi]
                    bdcondition = (obserr1 > 1) | (obserr1 <= 0.0)
                    bderr, = np.where(bdcondition)
                    nbderr = np.sum(bderr)
                    gderr, = np.where(~bdcondition)
                    ngderr = np.sum(gderr)
                    if (nbderr > 0) & (ngderr > 1): obserr1[bderr]=np.median([obserr1[gderr]])
                    obserr1 = medfilt(obserr1,51)
                    ccferr = ccorrelate(template[lo:hi],spec[lo:hi],lag,obserr1)

                nsumgood += 1
            else:   # no good pixels
                ccf = np.zeros(lag.shape,float)
                if (errccf is True) | (nofit is False): ccferr=ccf


            if (ichip==0):
                xcorr = ccf 
                if (ccferr is True): xcorrerr = ccferr^2
                tout = template[lo:hi] #+1
                sout = spec[lo:hi]     #+1
                errout = err[lo:hi]
            else:
                xcorr += ccf
                if (ccferr is True): xcorrerr += ccferr^2  # add in quadrature
                tout = [tout,template[lo:hi]] #+1
                sout = [sout,spec[lo:hi]]     # +1
                errout = [errout,err[lo:hi]]

        # No good cross-correlation
        if nsumgood==0:
            raise Exception("No good cross-correlation")

        if nsumgood>1: xcorr /= nsumgood   # want to average XCORR arrays from multiple chips
        if (xcorrerr is True):
            xcorrerr = np.sqrt(xcorrerr)
            if nsumgood>1: xcorrerr /= nsumgood

        # Remove the median
        xcorr -= np.median(xcorr)

        # Best shift
        best_shiftind0 = np.argmax(xcorr)
        best_xshift0 = lag[best_shiftind0]
        temp = np.roll( tout, best_xshift0)

        # Find Chisq for each synthetic spectrum
        gdpix, = np.where( (np.isfinite(sout)==1) & (np.isfinite(temp)==1) & (sout>0.0) & (errout>0.0) & (errout < 1e5))
        ngdpix = np.sum(gdpix)
        if (ngdpix==0):
            raise Exception("Bad spectrum")

        chisq = np.sqrt( np.sum( (sout[gdpix]-temp[gdpix])**2/errout[gdpix]**2 )/ngdpix )

        outstr["chisq"] = chisq
        outstr["ccf"] = xcorr
        if (errccf is True): outstr[i].ccferr = xcorrerr

        if (nofit is True): return

        # Remove smooth background at large scales
        cont = gsmooth(xcorr,100)
        xcorr_diff = xcorr-cont

        # Fit Xcorr peak with a Gaussian plus a line
        #---------------------------------------------
        # Some CCF peaks are SOOO wide that they span the whole width
        # do the first one without background subtraction
        estimates0 = [xcorr_diff[best_shiftind0],best_xshift0,4.0,0.0]
        lbounds0 = [1e-3, np.min(lag), 0.1, -2*np.max(xcorr_diff)]
        ubounds0 = [np.max(xcorr_diff)*2, np.max(lag), np.max(lag), 2*np.max(xcorr_diff) ]
        pars0, cov0 = gaussfit(lag,xcorr_diff,estimates0,xcorrerr,bounds=(lbounds0,ubounds0),binned=True)
        yfit0 = gaussbin(lag,*pars0)
        # Find peak in the background subtracted XCOR
        best_shiftind = np.argmax(xcorr_diff)
        best_xshift = lag[best_shiftind]
  
        # Fit the width
        #  keep height, center and constant constrained
        estimates = pars0
        estimates[1] = best_xshift
        lbounds0 = [0.5*estimates[0], best_xshift-4, 0.3*pars0[2], lt(lt(np.min(xcorr_diff),0),(pars0[3]-0.1)) ]
        ubounds0 = [1.5*estimates[0], best_xshift+4, 1.5*pars0[2],  gt(np.max(xcorr_diff)*0.5,(pars0[3]+0.1)) ]
        lo1 = gt(np.floor( best_shiftind-gt(pars0[2]*2,5)), 0)
        hi1 = lt(np.ceil( best_shiftind+gt(pars0[2]*2,5) ), (len(lag)-1))
        pars1, cov1 = gaussfit(lag[lo1:hi1],xcorr_diff[lo1:hi1],estimates,xcorrerr[lo1:hi1],
                               bounds=(lbounds1,ubounds1),binned=True)
        yfit1 = gaussbin(lag[lo1:hi1],*pars1)

        # Refit and let constant vary more, keep width constrained
        lo2 = gt(np.floor( best_shiftind-gt(pars1[2]*2,5)), 0)
        hi2 = lt(np.ceil( best_shiftind+gt(pars1[2]*2,5)), (len(lag)-1))
        est2 = pars1
        est2[3] = np.median(xcorr_diff[lo1:hi1]-yfit1) + pars1[3]
        lbounds2 = [0.5*pars1[0], limit(np.min(lag), (best_xshift-gt(pars1[2],1)), (pars1[1]-1)),
                    0.3*pars1[2], lt(lt(np.min(xcorr_diff),0), (est2[3]-0.1))]
        ubounds2 = [1.5*pars1[0], limit(np.max(lag), (pars1[1]+1), (best_xshift+gt(pars1[2],1))),
                    1.5*pars1[2], gt(np.max(xcorr_diff)*0.5, (est2[3]+0.1))]
        pars2, cov2 = gaussfit(lag[lo2:hi2],xcorr_diff[lo2:hi2],est2,xcorrerr[lo2:hi2],
                               bounds=(lbounds2,ubounds2),binned=True)
        yfit2 = gaussbin(lag[lo2:hi2],*pars2)
        
        # Refit with even narrower range
        lo2 = gt(np.floor( best_shiftind-gt(pars2[2],5)), 0)
        hi2 = lt(np.ceil( best_shiftind+gt(pars2[2],5)), (len(lag)-1))
        lbounds3 = [0.5*pars2[0], limit(np.min(lag), (best_xshift-gt(pars2[2],1)), (pars2[1]-1)),
                    0.3*pars2[2], lt(lt(np.min(xcorr_diff),0), (pars2[3]-0.1))]
        ubounds3 = [1.5*pars2[0], limit(np.max(lag), (pars2[1]+1), (best_xshift+gt(pars2[2],1))),
                    1.5*pars2[2], gt(np.max(xcorr_diff)*0.5, (pars2[3]+0.1))]
        pars3, cov3 = gaussfit(lag[lo3:hi3],xcorr_diff[lo3:hi3],pars2,xcorrerr[lo3:hi3],
                               bounds=(lbounds3,ubounds3),binned=True)
        yfit3 = gaussbin(lag[lo3:hi3],*pars3)

        # Final parameters
        pars = pars3
        perror = np.sqrt(np.diag(cov3))
        xshift = pars[1]
        xshifterr = perror[1]
        ccpfwhm_pix = pars[2]*2.35482  # ccp fwhm in pixels
        # v = (10^(delta log(wave))-1)*c
        dwlog = np.median(slope(np.log10(wave)))
        ccpfwhm = ( 10**(ccpfwhm_pix*dwlog)-1 )*cspeed  # in km/s

        # Convert pixel shift to velocity
        #---------------------------------
        # delta log(wave) = log(v/c+1)
        # v = (10^(delta log(wave))-1)*c
        dwlog = np.median(slope(np.log10(wave)))
        vrel = ( 10**(xshift*dwlog)-1 )*cspeed
        # Vrel uncertainty
        dvreldshift = np.log10(10.0)*(10**(xshift*dwlog))*dwlog*cspeed  # derivative wrt shift
        vrelerr = dvreldshift * xshifterr

        # Make XCORR structure and add to STR
        #------------------------------------
        outstr["xshift0"] = best_xshift
        outstr["ccp0"] = np.max(xcorr)
        outstr["xshift"] = xshift
        outstr["xshifterr"] = xshifterr
        #outstr["xshift_interp"] = xshift_interp
        outstr["ccpeak"] = pars[0] 
        outstr["ccpfwhm"] = ccpfwhm  # in km/s
        outstr["ccp_pars"] = pars
        outstr["ccp_perror"] = perror
        #outstr["ccp_polycoef"] = polycoef
        outstr["vrel"] = vrel
        outstr["vrelerr"] = vrelerr
        outstr["w0"] = np.min(wave)
        outstr["dw"] = dwlog
        #outstr["chisq"] = chisq
        
        # Printing
        #----------
        print(' Best Xshift = '+str(best_xshift)+' pixels')
        print(' Best Gaussian fit Xshift = '+str(xshift)+' +/- '+str(xshifterr)+' pixels')
        print(' Vrel = '+str(vre)+' +/- '+str(vrelerr)+' km/s')


    # Auto-correlation
    auto = ccorrelate(template,template,lag)

