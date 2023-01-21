#!/usr/bin/env python

"""READER.PY - spectral reader

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20210605'  # yyyymmdd

import os
import numpy as np
import warnings
import astropy
from astropy.io import fits
from astropy.table import Table
from scipy.ndimage.filters import median_filter
from dlnpyutils import utils as dln, bindata
import dill as pickle
import functools
from . import spec1d,utils
from .spec1d import Spec1D
try:
    import __builtin__ as builtins # Python 2
except ImportError:
    import builtins # Python 3
    
# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# Get print function to be used locally, allows for easy logging
print = utils.getprintfunc() 

astropy.io.fits.Conf.use_memmap = False  # load into memory

# Check the data
def datacheck(spec=None):
    """ Check that there aren't any bad values in FLUX or ERR."""
    if spec is None:
        return
    nanflux = np.sum(~np.isfinite(spec.flux))
    if nanflux>0:
        print('WARNING: Spectrum has '+str(nanflux)+' NaN/Inf FLUX pixels')
    nanerr = np.sum(~np.isfinite(spec.err))
    if nanerr>0:
        print('WARNING: Spectrum has '+str(nanerr)+' NaN/Inf ERR pixels')
    zeroerr = np.sum(spec.err<=0.0)
    if zeroerr>0:
        print('WARNING: Spectrum has '+str(zeroerr)+' pixels with ERR<=0')
    if np.sum(~spec.mask)==0:
        print('WARNING: Entire Spectrum is MASKED')
        

# Register a new reader
def register(format,reader):
    '''
    This registers a new spectral reader for the Doppler reader registry.

    Parameters
    ----------
    format : str
        The name of the new format.
    reader : function
        The function for the new reader.

    Returns
    -------
    Nothing is returned.  The Doppler reader registry is updated.

    Example
    -------
    doppler.reader.register('myformat',myformat)

    '''

    _readers[format] = reader

    
    
# Load a spectrum
def read(filename=None,format=None):
    '''
    This reads in a SDSS-IV MWM training set spectrum and returns an
    object that is guaranteed to have certain information.

    Parameters
    ----------
    filename : str
          The filename of the spectrum.
    format : str, optional
          The format string, or name of the reader to use.

    Returns
    -------
    spec : Spec1D object
        A Spec1D object that always has FLUX, ERR, WAVE, MASK, FILENAME, SPTYPE, WAVEREGIME and INSTRUMENT.

    Examples
    --------

    Load an APOGEE apStar spectrum.

    .. code-block:: python

        spec = read("apStar-r8-2M00050083+6349330.fits")

    '''
    if filename is None:
        print("Please input the filename")
        return

    # Different types of spectra
    # apVisit APOGEE spectra
    # apStar APOGEE spectra
    # SDSS BOSS spectra
    # SDSS MaStar spectra
    # IRAF-style spectra
    base, ext = os.path.splitext(os.path.basename(filename))

    # Check that the files exists
    if os.path.exists(filename) is False:
        print(filename+" NOT FOUND")
        return None
    
    # Readers dictionary
    try:
        nreaders = len(_readers)
    except:
        # _readers is defined at bottom of readers.py
        raise ValueError("No readers defined")
    # Format input
    if format is not None:
        # check that we have this reader
        if format.lower() not in _readers.keys():
            raise ValueError('reader '+format+' not found')
        # Use the requested reader/format
        out = _readers[format](filename)
        datacheck(out)
        if out is not None: return out
    # Check if it's spec1d
    else:
        head = fits.getheader(filename)
        if head.get('SPECTYPE')=='SPEC1D':
            out = spec1d(filename)
            datacheck(out)
            if out is not None: return out
        
    # Loop over all readers until we get a spectrum out
    for k in _readers.keys():
        try:
            out = _readers[k](filename)
        except:
            out = None
        datacheck(out)            
        if out is not None: return out
    # None found
    print('No reader recognized '+filename)
    print('Current list of readers: '+', '.join(_readers.keys()))
    return


# read spectrum written with Spec1D.write()
def spec1d(filename):
    """
    Read a Doppler written spectrum.

    Parameters
    ----------
    filename : string
         The name of the spectrum file to load.

    Returns
    -------
    spec : Spec1D object
       The spectrum as a Spec1D object.

    Examples
    --------
    
    spec = spec1d('spec.fits')

    """

    base, ext = os.path.splitext(os.path.basename(filename))

    # Get number of extensions
    hdu = fits.open(filename)
    
    # Check that this has the right format
    if hdu[0].header['SPECTYPE'] != 'SPEC1D':
        return None

    # HDU0: header
    wavevac = hdu[0].header['WAVEVAC']
    instrument = hdu[0].header['INSTRMNT']
    normalized = hdu[0].header['NORMLIZD']
    ndim = hdu[0].header['NDIM']
    npix = hdu[0].header['NPIX']
    norder = hdu[0].header['NORDER']
    snr = float(hdu[0].header['SNR'])
    bc = hdu[0].header.get('BC')
    # HDU1: flux
    flux = hdu[1].data
    # HDU2: flux error
    err = hdu[2].data
    # HDU3: wavelength
    wave = hdu[3].data
    # HDU4: mask
    mask = hdu[4].data
    # HDU5: LSF
    lsfsigma = hdu[5].data
    lsfxtype = hdu[5].header['XTYPE']
    lsftype = hdu[5].header['LSFTYPE']
    npars = hdu[5].header['NPARS']
    npars1 = hdu[5].header['NPARS1']
    npars2 = hdu[5].header.get('NPARS2')
    if npars>0:
        lsfpars = np.zeros(npars,float)
        for i in range(npars):
            lsfpars[i] = hdu[5].header['PAR'+str(i)]
        if npars2 is not None:
            lsfpars = lsfpars.reshape(npars1,npars2)
        else:
            lsfpars = lsfpars.reshape(npars1)            
    # HDU6: continuum
    cont = hdu[6].data
    # HDU7: continuum function
    cont_func_tab = hdu[7].data

    # Create the Spec1D object
    spec = Spec1D(flux,wave=wave,lsfpars=lsfpars,lsftype=lsftype,lsfxtype=lsfxtype)
    spec.reader = 'spec1d'
    spec.filename = filename
    spec.err = err
    if mask is not None:
        spec.mask = mask.astype(bool)
    if cont is not None:
        spec._cont = cont
    spec.continuum_func = pickle.loads(cont_func_tab['func'][0])
    spec.instrument = instrument
    spec.wavevac = True
    if bc is not None: spec.bc = bc
    
    # Add extra attributes
    if len(hdu)>8:
        for i in np.arange(8,len(hdu)):
            val = hdu[i].data
            vector = hdu[i].header.get('vector')
            attribut = hdu[i].header.get('attribut')
            name = hdu[i].header.get('name')
            if attribut==True and vector==False:  # scalar table
                tab = hdu[i].data
                for c in tab.dtype.names:
                    val = tab[c][0]
                    if val=='None': val=None
                    setattr(spec,c,val)
            elif attribut==True and vector==True:  # vector
                setattr(spec,name,val)
    hdu.close()
    
    return spec


# Load APOGEE apVisit/asVisit spectra
def apvisit(filename):
    """
    Read a SDSS APOGEE apVisit spectrum.

    Parameters
    ----------
    filename : string
         The name of the spectrum file to load.

    Returns
    -------
    spec : Spec1D object
       The spectrum as a Spec1D object.

    Examples
    --------
    
    spec = apvisit('spec.fits')

    """

    base, ext = os.path.splitext(os.path.basename(filename))

    # Get number of extensions
    hdulist = fits.open(filename)
    nhdu = len(hdulist)
    hdulist.close()
    
    # Check that this has the right format
    validfile = False
    if (base.find("apVisit") == -1) | (base.find("asVisit") == -1):
        if nhdu>=10:
            gd, = np.where(np.char.array(hdulist[0].header['HISTORY']).astype(str).find('AP1DVISIT') > -1)
            if len(gd)>0: validfile=True
    if validfile is False:
        return None
        
    # APOGEE apVisit, visit-level spectrum

    # HISTORY AP1DVISIT:  HDU0 = Header only                                          
    # HISTORY AP1DVISIT:  HDU1 - Flux (10^-17 ergs/s/cm^2/Ang)                        
    # HISTORY AP1DVISIT:  HDU2 - Error (10^-17 ergs/s/cm^2/Ang)                       
    # HISTORY AP1DVISIT:  HDU3 - Flag mask (bitwise OR combined)                      
    # HISTORY AP1DVISIT:  HDU4 - Wavelength (Ang)                                     
    # HISTORY AP1DVISIT:  HDU5 - Sky (10^-17 ergs/s/cm^2/Ang)                         
    # HISTORY AP1DVISIT:  HDU6 - Sky Error (10^-17 ergs/s/cm^2/Ang)                   
    # HISTORY AP1DVISIT:  HDU7 - Telluric                                             
    # HISTORY AP1DVISIT:  HDU8 - Telluric Error                                       
    # HISTORY AP1DVISIT:  HDU9 - Wavelength coefficients                              
    # HISTORY AP1DVISIT:  HDU10 - LSF coefficients
    # HISTORY AP1DVISIT:  HDU11 - RV catalog

    # flux, err, sky, skyerr are in units of 1e-17
    flux = fits.getdata(filename,1).T * 1e-17   # [Npix,Norder]
    wave = fits.getdata(filename,4).T
    lsfcoef = fits.getdata(filename,10).T
    spec = Spec1D(flux,wave=wave,lsfpars=lsfcoef,lsftype='Gauss-Hermite',lsfxtype='Pixels')
    spec.reader = 'apvisit'
    spec.filename = filename
    spec.sptype = "apVisit"
    spec.waveregime = "NIR"
    spec.instrument = "APOGEE"        
    spec.head = fits.getheader(filename,0)
    spec.err = fits.getdata(filename,2).T * 1e-17  # [Npix,Norder]
    #bad = (spec.err<=0)   # fix bad error values
    #if np.sum(bad) > 0:
    #    spec.err[bad] = 1e30
    spec.bitmask = fits.getdata(filename,3).T
    spec.sky = fits.getdata(filename,5).T * 1e-17
    spec.skyerr = fits.getdata(filename,6).T * 1e-17
    spec.telluric = fits.getdata(filename,7).T
    spec.telerr = fits.getdata(filename,8).T
    spec.wcoef = fits.getdata(filename,9).T        
    # Create the bad pixel mask
    # "bad" pixels:
    #   flag = ['BADPIX','CRPIX','SATPIX','UNFIXABLE','BADDARK','BADFLAT','BADERR','NOSKY',
    #           'LITTROW_GHOST','PERSIST_HIGH','PERSIST_MED','PERSIST_LOW','SIG_SKYLINE','SIG_TELLURIC','NOT_ENOUGH_PSF','']
    #   badflag = [1,1,1,1,1,1,1,1,
    #              0,0,0,0,0,0,1,0]
    mask = (np.bitwise_and(spec.bitmask,16639)!=0) | (np.isfinite(spec.flux)==False)
    # Extra masking for bright skylines
    x = np.arange(spec.npix)
    nsky = 4
    for i in range(spec.norder):
        sky = spec.sky[:,i]
        medsky = median_filter(sky,201,mode='reflect')
        medcoef = dln.poly_fit(x,medsky/np.median(medsky),2)
        medsky2 = dln.poly(x,medcoef)*np.median(medsky)
        skymask1 = (sky>nsky*medsky2)    # pixels Nsig above median sky
        mask[:,i] = np.logical_or(mask[:,i],skymask1)    # OR combine
    spec.mask = mask
    # Fix NaN pixels
    for i in range(spec.norder):
        bd,nbd = dln.where( (np.isfinite(spec.flux[:,i])==False) | (spec.err[:,i] <= 0) )
        if nbd>0:
            spec.flux[bd,i] = 0.0
            spec.err[bd,i] = 1e30
            spec.mask[bd,i] = True
    if (nhdu>=11):
        spec.meta = fits.getdata(filename,11)   # catalog of RV and other meta-data
    # Spectrum, error, sky, skyerr are in units of 1e-17
    #spec.snr = spec.head["SNR"]
    if base.find("apVisit") > -1:
        spec.observatory = 'apo'
    else:
        spec.observatory = 'lco'
    spec.wavevac = True
    return spec


# Load APOGEE apStar/asStar spectra    
def apstar(filename):
    """
    Read an SDSS APOGEE apStar spectrum.

    Parameters
    ----------
    filename : string
         The name of the spectrum file to load.

    Returns
    -------
    spec : Spec1D object
       The spectrum as a Spec1D object.

    Examples
    --------
    
    spec = apstar('spec.fits')

    """

    base, ext = os.path.splitext(os.path.basename(filename))

    # Get number of extensions
    hdulist = fits.open(filename)
    nhdu = len(hdulist)
    hdulist.close()
    
    # Check that this has the right format
    validfile = False
    if (base.find("apStar") == -1) | (base.find("asStar") == -1):
        if nhdu>=9:
            gd, = np.where(np.char.array(hdulist[0].header['HISTORY']).astype(str).find('APSTAR') > -1)
            if len(gd)>0: validfile=True
    if validfile is False:
        return None
    
    
    # APOGEE apStar, combined spectrum

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

    # Spectrum, error, sky, skyerr are in units of 1e-17
    #  these are 2D arrays with [Nvisit+2,Npix]
    #  the first two are combined and the rest are the individual spectra

    head1 = fits.getheader(filename,1)
    w0 = np.float64(head1["CRVAL1"])
    dw = np.float64(head1["CDELT1"])
    nw = head1["NAXIS1"]
    wave = 10**(np.arange(nw)*dw+w0)
        
    # flux, err, sky, skyerr are in units of 1e-17
    flux = fits.getdata(filename,1).T * 1e-17
    lsfcoef = fits.getdata(filename,8).T
    spec = Spec1D(flux,wave=wave,lsfpars=lsfcoef,lsftype='Gauss-Hermite',lsfxtype='Pixels')
    spec.reader = 'apstar'
    spec.filename = filename
    spec.sptype = "apStar"
    spec.waveregime = "NIR"
    spec.instrument = "APOGEE"
    spec.head = fits.getheader(filename,0)
    spec.err = fits.getdata(filename,2).T * 1e-17
    #bad = (spec.err<=0)   # fix bad error values
    #if np.sum(bad) > 0:
    #    spec.err[bad] = 1e30
    spec.bitmask = fits.getdata(filename,3)
    spec.sky = fits.getdata(filename,4).T * 1e-17
    spec.skyerr = fits.getdata(filename,5).T * 1e-17
    spec.telluric = fits.getdata(filename,6).T
    spec.telerr = fits.getdata(filename,7).T
    spec.lsf = fits.getdata(filename,8).T
    # Create the bad pixel mask
    # "bad" pixels:
    #   flag = ['BADPIX','CRPIX','SATPIX','UNFIXABLE','BADDARK','BADFLAT','BADERR','NOSKY',
    #           'LITTROW_GHOST','PERSIST_HIGH','PERSIST_MED','PERSIST_LOW','SIG_SKYLINE','SIG_TELLURIC','NOT_ENOUGH_PSF','']
    #   badflag = [1,1,1,1,1,1,1,1,
    #              0,0,0,0,0,0,1,0]
    mask = (np.bitwise_and(spec.bitmask,16639)!=0) | (np.isfinite(spec.flux)==False)
    # Extra masking for bright skylines
    x = np.arange(spec.npix)
    nsky = 4
    medsky = median_filter(spec.sky,201,mode='reflect')
    medcoef = dln.poly_fit(x,medsky/np.median(medsky),2)
    medsky2 = dln.poly(x,medcoef)*np.median(medsky)
    skymask1 = (sky>nsky*medsky2)    # pixels Nsig above median sky
    mask[:,i] = np.logical_or(mask[:,i],skymask1)    # OR combine
    spec.mask = mask
    # Fix NaN or bad pixels pixels
    bd,nbd = dln.where( (np.isfinite(spec.flux[:,i])==False) | (spec.err[:,i] <= 0.0) )
    if nbd>0:
        spec.flux[bd] = 0.0
        spec.err[bd] = 1e30
        spec.mask[bd] = True
    if nhdu>=9:
        spec.meta = fits.getdata(filename,9)    # meta-data
    #spec.snr = spec.head["SNR"]
    if base.find("apStar") > -1:
        spec.observatory = 'apo'
    else:
        spec.observatory = 'lco'
    spec.wavevac = True            
    return spec


# Load SDSS BOSS spectra    
def boss(filename):
    """
    Read a SDSS BOSS spectrum.

    Parameters
    ----------
    filename : string
         The name of the spectrum file to load.

    Returns
    -------
    spec : Spec1D object
       The spectrum as a Spec1D object.

    Examples
    --------
    
    spec = boss('spec.fits')

    """

    base, ext = os.path.splitext(os.path.basename(filename))

    # Get number of extensions
    hdulist = fits.open(filename)
    nhdu = len(hdulist)
    hdulist.close()
    
    # Check that this has the right format
    validfile = False
    if (base.find("spec-") == -1) | (base.find("SDSS") == -1):
        # nothing definitive in the header
        if nhdu>=3:
            validfile = True
    if validfile is False:
        return None
    
    # BOSS spec

    # HDU1 - binary table of spectral data
    # HDU2 - table with metadata including S/N
    # HDU3 - table with line measurements
    head = fits.getheader(filename,0)
    tab1 = Table.read(filename,1)
    cat1 = Table.read(filename,2)
    flux = tab1["flux"].data
    wave = 10**tab1["loglam"].data
    wdisp = tab1["wdisp"].data
    # checking for zeros in IVAR
    ivar = tab1["ivar"].data.copy()
    bad = (ivar<=0)
    if np.sum(bad) > 0:
        ivar[bad] = 1.0
        err = 1.0/np.sqrt(ivar)
        err[bad] = 1e30  #np.nan
        mask = np.zeros(flux.shape,bool)
        mask[bad] = True
    else:
        err = 1.0/np.sqrt(ivar)
        mask = np.zeros(flux.shape,bool)
    spec = Spec1D(flux,err=err,wave=wave,mask=mask,lsfsigma=wdisp,lsfxtype='Wave')
    spec.reader = 'boss'
    spec.lsf.clean()   # clean up some bad LSF values
    spec.filename = filename
    spec.sptype = "spec"
    spec.waveregime = "Optical"
    spec.instrument = "BOSS"
    spec.head = head
    spec.ivar = tab1["ivar"].data
    spec.bitmask = tab1["or_mask"].data
    spec.and_mask = tab1["and_mask"].data
    spec.or_mask = tab1["or_mask"].data
    spec.sky = tab1["sky"].data
    spec.model = tab1["model"].data
    spec.meta = cat1
    # What are the units?
    #spec.snr = cat1["SN_MEDIAN_ALL"].data[0]
    spec.observatory = 'apo'
    spec.wavevac = True
    return spec        

    
# Load SDSS MaStar spectra
def mastar(filename):
    """
    Read a SDSS MaStar spectrum.

    Parameters
    ----------
    filename : string
         The name of the spectrum file to load.

    Returns
    -------
    spec : Spec1D object
       The spectrum as a Spec1D object.

    Examples
    --------
    
    spec = mastar('spec.fits')

    """

    base, ext = os.path.splitext(os.path.basename(filename))

    # Get number of extensions
    hdulist = fits.open(filename)
    nhdu = len(hdulist)
    hdulist.close()
    
    # Check that this has the right format
    validfile = False
    if (base.find("mastar-") == -1):
        # Not much definitive in the header
        if nhdu>=1:
            validfile = True
    if validfile is False:
        return None
    
    # MaStar spec

    # HDU1 - table with spectrum and metadata
    tab = Table.read(filename,1)
    flux = tab["FLUX"].data[0]
    # checking for zeros in IVAR
    ivar = tab["IVAR"].data[0].copy()
    bad = (ivar<=0)
    if np.sum(bad) > 0:
        ivar[bad] = 1.0
        err = 1.0/np.sqrt(ivar)
        err[bad] = 1e30 #np.nan
        mask = np.zeros(flux.shape,bool)
        mask[bad] = True
    else:
        err = 1.0/np.sqrt(ivar)
        mask = np.zeros(flux.shape,bool)
    spec = Spec1D(flux,wave=wave,err=err,mask=mask)
    spec.reader = 'mastar'
    spec.filename = filename
    spec.sptype = "MaStar"
    spec.waveregime = "Optical"
    spec.instrument = "BOSS"        
    spec.ivar = tab["IVAR"].data[0]
    spec.wave = tab["WAVE"].data[0]
    spec.bitmask = tab["MASK"].data[0]
    spec.disp = tab["DISP"].data[0]
    spec.presdisp = tab["PREDISP"].data[0]
    meta = {'DRPVER':tab["DRPVER"].data,'MPROCVER':tab["MPROCVER"].data,'MANGAID':tab["MANGAID"].data,'PLATE':tab["PLATE"].data,
            'IFUDESIGN':tab["IFUDESIGN"].data,'MJD':tab["MJD"].data,'IFURA':tab["IFURA"].data,'IFUDEC':tab["IFUDEC"].data,
            'OBJRA':tab["OBJRA"].data,'OBJDEC':tab["OBJDEC"].data,'PSFMAG':tab["PSFMAG"].data,'MNGTARG2':tab["MNGTARG2"].data,
            'NEXP':tab["NEXP"].data,'HELIOV':tab["HELIOV"].data,'VERR':tab["VERR"].data,'V_ERRCODE':tab["V_ERRCODE"].data,
            'MJDQUAL':tab["MJDQUAL"].data,'SNR':tab["SNR"].data,'PARS':tab["PARS"].data,'PARERR':tab["PARERR"].data}
    spec.meta = meta
    # What are the units?
    #spec.snr = tab["SNR"].data
    spec.observatory = 'apo'
    spec.wavevac = True        
    return spec        

    
# Load IMACS spectra
def imacs(filename):
    """
    Read IMACS spectrum.

    Parameters
    ----------
    filename : string
         The name of the spectrum file to load.

    Returns
    -------
    spec : Spec1D object
       The spectrum as a Spec1D object.

    Examples
    --------
    
    spec = imacs('spec.fits')

    """

    base, ext = os.path.splitext(os.path.basename(filename))
    
    # Generic IRAF spectrum
    # BANDID1 = 'spectrum: background median, weights variance, clean yes' /          
    # BANDID2 = 'sigma - background median, weights variance, clean yes' /            
    # BANDID3 = 'mask - 0:good, 1:bad' / 
    data,head = fits.getdata(filename,0,header=True)
    ndim = data.ndim
    # often the 2nd dimension is unnecessary, e.g. [3, 1, 1649]
    if (ndim==3):
        if data.shape[1]==1: data = data[:,0,:]
        ndim = data.ndim
    flux = data[0,:]
    npix = len(flux)    
    sigma = data[1,:]
    badmask = data[2,:]
    mask = np.zeros(npix,bool)
    mask[badmask==1] = True
    # Get wavelength
    # NWPAR   =                    4 / Number of Wavelength solution parameters       
    # WPAR1   =        4301.64021493 / Wavelength solution parameter                  
    # WPAR2   =        1.70043816611 / Wavelength solution parameter                  
    # WPAR3   =   -3.92469453122E-06 / Wavelength solution parameter                  
    # WPAR4   =    1.99202801624E-09 / Wavelength solution parameter                  
    # WSIG    =       0.125175450918 / Sigma of wavelenth solution in Ang             
    # WCSDIM  =                    3 /    
    nwpar = head['nwpar']
    wpar = np.zeros(nwpar,np.float64)
    for i in range(nwpar):
        wpar[i] = head['WPAR'+str(i+1)]
    wpar = wpar[::-1]    # reverse
    wave = np.poly1d(wpar)(np.arange(npix))

    spec = Spec1D(flux,err=sigma,wave=wave,mask=mask)
    spec.reader = 'imacs'
    spec.filename = filename
    spec.sptype = "IMACS"
    spec.head = head
    spec.observatory = 'LCO'
    spec.wavevac = False
    
    return spec

# Load HYDRA spectra
def hydra(filename):
    """
    Read HYDRA spectrum.

    Parameters
    ----------
    filename : string
         The name of the spectrum file to load.

    Returns
    -------
    spec : Spec1D object
       The spectrum as a Spec1D object.

    Examples
    --------
    
    spec = hydra('spec.fits')

    """

    base, ext = os.path.splitext(os.path.basename(filename))
    
    # Generic IRAF spectrum
    # BANDID1 = 'spectrum: background median, weights variance, clean yes' /          
    # BANDID2 = 'sigma - background median, weights variance, clean yes' /            
    # BANDID3 = 'mask - 0:good, 1:bad' / 
    data,head = fits.getdata(filename,0,header=True)
    ndim = data.ndim
    # often the 2nd dimension is unnecessary, e.g. [3, 1, 1649]
    if (ndim==3):
        if data.shape[1]==1: data = data[:,0,:]
        ndim = data.ndim
    if (ndim==2):
        flux = data[0,:]
        npix = len(flux)            
        sigma = data[1,:]
        badmask = data[2,:]
        mask = np.zeros(npix,bool)
        mask[badmask==1] = True        
    elif (ndim==1):
        flux = data
        npix = len(flux)        
        sigma = np.sqrt(np.maximum(flux,1))  # assume gain~1
        mask = np.zeros(npix,bool)
    # Get wavelength
    crval1 = head.get('CRVAL1')
    crpix1 = head.get('CRPIX1')
    cdelt1 = head.get('CDELT1')
    if cdelt1 is None: cdelt1=head.get('CD1_1')
    if (crval1 is not None) & (crpix1 is not None) & (cdelt1 is not None):
        wave = (np.arange(npix)+1-crpix1) * np.float64(cdelt1) + np.float64(crval1)
        wlogflag = head.get('DC-FLAG')
        if (wlogflag is not None):
            if wlogflag==1: wave = 10**wave
    else:
        print('No wavelength information')
        wave = None

    spec = Spec1D(flux,err=sigma,wave=wave,mask=mask)
    spec.reader = 'hydra'
    spec.filename = filename
    spec.sptype = "HYDRA"
    spec.head = head
    spec.wavevac = False
    observat = head.get('OBSERVAT')
    if observat is not None:
        spec.observatory = observat.lower()

    # Deal with bad pixels
    bdpix, = np.where(np.isfinite(spec.flux)==False)
    if len(bdpix)>0:
        spec.flux[bdpix] = 0
        spec.err[bdpix] = 1e30
        spec.mask[bdpix] = True
        
    # Modify continuum parameters
    spec.continuum_func = functools.partial(spec1d.continuum,norder=4,perclevel=75.0,binsize=0.15,interp=True)
    # Mask last 100 pixels and first 75 that are below 0.8 of the continuum
    try:
        cont = spec.continuum_func(spec)
        x = np.arange(spec.npix)
        bd, = np.where((spec.flux/cont < 0.85) & ((x > (spec.npix-100)) | (x<75)))
        if len(bd)>0:
            spec.mask[bd] = True
            spec.err[bd] = 1e30
    except:
        spec.mask[0:100] = True
        spec.err[0:100] = 1e30
        spec.mask[-100:] = True
        spec.err[-100:] = 1e30        
        
    return spec

    
# Load IRAF-style spectra
def iraf(filename):
    """
    Read an IRAF-style spectrum.

    Parameters
    ----------
    filename : string
         The name of the spectrum file to load.

    Returns
    -------
    spec : Spec1D object
       The spectrum as a Spec1D object.

    Examples
    --------
    
    spec = iraf('spec.fits')

    """

    base, ext = os.path.splitext(os.path.basename(filename))
    
    # Generic IRAF spectrum
    #BANDID1 = 'spectrum: background fit, weights variance, clean no'                
    #BANDID2 = 'spectrum: background fit, weights none, clean no'                    
    #BANDID3 = 'background: background fit'                                          
    #BANDID4 = 'sigma - background fit, weights variance, clean no'  
    data,head = fits.getdata(filename,0,header=True)
    ndim = data.ndim
    if ndim==2: 
        npix,norder = data.shape
        flux = data[:,0]
    else:
        flux = data
        norder = 1
        npix = len(flux)
    # Get wavelength
    crval1 = head.get('crval1')
    crpix1 = head.get('crpix1')
    cdelt1 = head.get('cdelt1')
    if cdelt1 is None: cdelt1=head.get('cd1_1')
    if (crval1 is not None) & (crpix1 is not None) & (cdelt1 is not None):
        wave = (np.arange(npix,int)+1-crpix1) * np.float64(cdelt1) + np.float64(crval1)
        wlogflag = head.get('DC-FLAG')
        if (wlogflag is not None):
            if wlogflag==1: wave = 10**wave
    else:
        print('No wavelength information')
        wave = None
    spec = Spec1D(flux,wave=wave)
    spec.reader = 'iraf'
    spec.filename = filename
    spec.sptype = "IRAF"
    spec.head = head
    # Error/sigma array
    bandid = head.get('BANDID*')
    if bandid is not None:
        for i,key in enumerate(bandid):
            val = bandid[key]
            if 'sigma' in val:
                if (i<=norder):
                    spec.err = data[:,k]

    return spec

# List of readers
_readers = {'apvisit':apvisit, 'apstar':apstar, 'boss':boss, 'mastar':mastar, 'iraf':iraf,
            'spec1d':spec1d, 'imacs':imacs, 'hydra':hydra}
