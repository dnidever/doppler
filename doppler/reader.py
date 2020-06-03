#!/usr/bin/env python

"""READER.PY - spectral reader

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@noao.edu>'
__version__ = '20200111'  # yyyymmdd                                                                                                                           

import os
import numpy as np
import warnings
from astropy.io import fits
from astropy.table import Table
from scipy.ndimage.filters import median_filter
from dlnpyutils import utils as dln, bindata
from .spec1d import Spec1D
import matplotlib.pyplot as plt
import pdb

# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


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
        # _readers is defined in __init__.py
        raise ValueError("No readers defined")
    # Format input
    if format is not None:
        # check that we have this reader
        if format.lower() not in _readers.keys():
            raise ValueError('reader '+format+' not found')
        # Use the requested reader/format
        out = _readers[format](filename)
        if out is not None: return out
        
    # Loop over all readers until we get a spectrum out
    for k in _readers.keys():
        try:
            out = _readers[k](filename)
        except:
            out = None
        if out is not None: return out
    # None found
    print('No reader recognized '+filename)
    print('Current list of readers: '+', '.join(_readers.keys()))
    return


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
    
    # APOGEE apVisit, visit-level spectrum
    if (base.find("apVisit") > -1) | (base.find("asVisit") > -1):
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
        # Get number of extensions
        hdulist = fits.open(filename)
        nhdu = len(hdulist)
        hdulist.close()

        # flux, err, sky, skyerr are in units of 1e-17
        flux = fits.getdata(filename,1).T * 1e-17   # [Npix,Norder]
        wave = fits.getdata(filename,4).T
        lsfcoef = fits.getdata(filename,10).T
        spec = Spec1D(flux,wave=wave,lsfpars=lsfcoef,lsftype='Gauss-Hermite',lsfxtype='Pixels')
        spec.filename = filename
        spec.sptype = "Visit"
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
        #mask = (np.bitwise_and(spec.bitmask,16639)!=0) | (np.isfinite(spec.flux)==False)
        mask = (np.bitwise_and(spec.bitmask,20735)!=0) | (np.isfinite(spec.flux)==False)
        # Extra masking for bright skylines
        x = np.arange(spec.npix)
        nsky = 4
        #plt.clf()
        for i in range(spec.norder):
            sky = spec.sky[:,i]
            medsky = median_filter(sky,201,mode='reflect')
            medcoef = dln.poly_fit(x,medsky/np.median(medsky),2)
            medsky2 = dln.poly(x,medcoef)*np.median(medsky)
            skymask1 = (sky>nsky*medsky2)    # pixels Nsig above median sky
            #mask[:,i] = np.logical_or(mask[:,i],skymask1)    # OR combine
            #plt.plot(spec.wave[:,i],sky)
            #plt.plot(spec.wave[:,i],nsky*medsky2)
            #plt.plot(spec.wave[:,i],spec.flux[:,i])
        #plt.draw()
        #pdb.set_trace()
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
        spec.snr = spec.head["SNR"]
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
    
    # APOGEE apStar, combined spectrum
    if (base.find("apStar") > -1)  | (base.find("asStar") > -1):
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
        # Get number of extensions
        hdulist = fits.open(filename)
        nhdu = len(hdulist)
        hdulist.close()

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
        spec.snr = spec.head["SNR"]
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
    
    # BOSS spec
    if (base.find("spec-") > -1) | (base.find("SDSS") > -1):
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
        spec.snr = cat1["SN_MEDIAN_ALL"].data[0]
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
    
    # MaStar spec
    if (base.find("mastar-") > -1):
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
        spec.snr = tab["SNR"].data
        spec.observatory = 'apo'
        spec.wavevac = True        
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
    if (crval1 is not None) & (crpix1 is not None) & (cdelt is not None):
        wave = (np.arange(npix,int)+1-crpix1) * np.float64(cdelt1) + np.float64(crval1)
        wlogflag = head.get('DC-FLAG')
        if (wlogflag is not None):
            if wlogflag==1: wave = 10**wave
    else:
        print('No wavelength information')
        wave = None
    spec = Spec1D(flux,wave=wave)
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
_readers = {'apvisit':apvisit, 'apstar':apstar, 'boss':boss, 'mastar':mastar, 'iraf':iraf}
