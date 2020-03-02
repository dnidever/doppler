#!/usr/bin/env python

"""UTILS.PY - Utility functions

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@noao.edu>'
__version__ = '20200112'  # yyyymmdd                                                                                                                           

import os
import numpy as np
import warnings
from scipy import sparse
from scipy.interpolate import interp1d
from dlnpyutils import utils as dln
import matplotlib.pyplot as plt

# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

cspeed = 2.99792458e5  # speed of light in km/s


# Convert wavelengths to pixels for a dispersion solution
def w2p(dispersion,w,extrapolate=True):
    """
    Convert wavelength values to pixels for a "dispersion solution".

    Parameters
    ----------
    dispersion : 1D array
         The dispersion solution.  This is basically just a 1D array of
         monotonically increasing (or decreasing) wavelengths.
    w : array
      Array of wavelength values to convert to pixels.
    extrapolate : bool, optional
       Extrapolate beyond the dispersion solution, if necessary.
       This is True by default.

    Returns
    -------
    x : array
      Array of converted pixel values.

    Examples
    --------
    x = w2p(disp,w)

    """

    x = interp1d(dispersion,np.arange(len(dispersion)),kind='cubic',bounds_error=False,fill_value=(np.nan,np.nan),assume_sorted=False)(w)
    # Need to extrapolate
    if ((np.min(w)<np.min(dispersion)) | (np.max(w)>np.max(dispersion))) & (extrapolate is True):
        win = dispersion
        xin = np.arange(len(dispersion))
        si = np.argsort(win)
        win = win[si]
        xin = xin[si]
        npix = len(win)
        # At the beginning
        if (np.min(w)<np.min(dispersion)):
            coef1 = dln.poly_fit(win[0:10], xin[0:10], 2)
            bd1, nbd1 = dln.where(w < np.min(dispersion))
            x[bd1] = dln.poly(w[bd1],coef1)
        # At the end
        if (np.max(w)>np.max(dispersion)):
            coef2 = dln.poly_fit(win[npix-10:], xin[npix-10:], 2)
            bd2, nbd2 = dln.where(w > np.max(dispersion))
            x[bd2] = dln.poly(w[bd2],coef2)                
    return x


# Convert pixels to wavelength for a dispersion solution
def p2w(dispersion,x,extrapolate=True):
    """
    Convert pixel values to wavelengths for a "dispersion solution".

    Parameters
    ----------
    dispersion : 1D array
         The dispersion solution.  This is basically just a 1D array of
         monotonically increasing (or decreasing) wavelengths.
    x : array
      Array of pixel values to convert to wavelengths.
    extrapolate : bool, optional
       Extrapolate beyond the dispersion solution, if necessary.
       This is True by default.

    Returns
    -------
    w : array
      Array of converted wavelengths.

    Examples
    --------
    w = p2w(disp,x)

    """

    npix = len(dispersion)
    w = interp1d(np.arange(len(dispersion)),dispersion,kind='cubic',bounds_error=False,fill_value=(np.nan,np.nan),assume_sorted=False)(x)
    # Need to extrapolate
    if ((np.min(x)<0) | (np.max(x)>(npix-1))) & (extrapolate is True):
        xin = np.arange(npix)
        win = dispersion
        # At the beginning
        if (np.min(x)<0):
            coef1 = dln.poly_fit(xin[0:10], win[0:10], 2)
            bd1, nbd1 = dln.where(x < 0)
            w[bd1] = dln.poly(x[bd1],coef1)
        # At the end
        if (np.max(x)>(npix-1)):
            coef2 = dln.poly_fit(xin[npix-10:], win[npix-10:], 2)
            bd2, nbd2 = dln.where(x > (npix-1))
            w[bd2] = dln.poly(x[bd2],coef2)                
    return w


def sparsify(lsf):
    """
    This is a helper function for convolve_sparse() that takes a 
    2D LSF array and returns a sparse matrix.

    Parameters
    ----------
    lsf : 2D array
         A 2D Line Spread Function (LSF) to sparsify.

    Returns
    -------
    out : sparse matrix
         The sparse matrix version of lsf.

    Usage
    -----
    slsf = sparsify(lsf)

    From J.Bovy's lsf.py APOGEE code.
    """

    nx = lsf.shape[1]
    diagonals = []
    offsets = []
    for ii in range(nx):
        offset= nx//2-ii
        offsets.append(offset)
        if offset < 0:
            diagonals.append(lsf[:offset,ii])
        else:
            diagonals.append(lsf[offset:,ii])
    return sparse.diags(diagonals,offsets)


def convolve_sparse(spec,lsf):
    """
    Convolve a flux array (1D) with an LSF (2D) using sparse matrices.

    Parameters
    ----------
    spec : 1D array
         The 1D flux array [Npix].
    lsf : 2D array
         The Line Spread Function (LSF) to convolve the spectrum with.  This must
         have shape of [Npix,Nlsf].

    Returns
    -------
    out : array
         The new flux array convolved with LSF.

    Usage
    -----
    >>>out = convolve_sparse(spec,lsf)

    From J.Bovy's lsf.py APOGEE code.
    """

    npix,nlsf = lsf.shape
    lsf2 = sparsify(lsf)
    spec2 = np.reshape(spec,(1,npix))
    spec2 = sparse.csr_matrix(spec2) 
    out = lsf2.dot(spec2.T).T.toarray()
    out = np.reshape(out,len(spec))
    # The ends are messed up b/c not normalized
    hlf = nlsf//2
    for i in range(hlf+1):
        lsf1 = lsf[i,hlf-i:]
        lsf1 /= np.sum(lsf1)
        out[i] = np.sum(spec[0:len(lsf1)]*lsf1)
    for i in range(hlf+1):
        ii = npix-i-1
        lsf1 = lsf[ii,:hlf+1+i]
        lsf1 /= np.sum(lsf1)
        out[ii] = np.sum(spec[npix-len(lsf1):]*lsf1)
    return out


# Make logaritmic wavelength scale
def make_logwave_scale(wave,vel=1000.0):
    """
    Create a logarithmic wavelength scale for a given wavelength array.

    This is used by rv.fit() to create a logarithmic wavelength scale for
    cross-correlation.

    If the input wavelength array is 2D with multiple orders (trailing
    dimension), then the output wavelength array will extend beyond the
    boundaries of some of the orders.  These pixels will have to be
    "padded" with dummy/masked-out pixels.

    Parameters
    ----------
    wave : array
         Input wavelength array, 1D or 2D with multiple orders.
    vel : float, optional
         Maximum velocity shift to allow for.  Default is 1000.

    Returns
    -------
    fwave : array
         New logarithmic wavelength array.

    Usage
    -----
    >>>fwave = make_logwave_scale(wave)

    """

    # If the existing wavelength scale is logarithmic then use it, just extend
    # on either side
    vel = np.abs(vel)
    wave = np.float64(wave)
    nw = len(wave)
    wr = dln.minmax(wave)

    # Multi-order wavelength array
    if wave.ndim==2:
        norder = wave.shape[1]

        # Ranges
        wr = np.empty((2,norder),np.float64)
        wlo = np.empty(norder,np.float64)
        whi = np.empty(norder,np.float64)
        for i in range(norder):
            wr[:,i] = dln.minmax(wave[:,i])
            wlo[i] = wr[0,i]-vel/cspeed*wr[0,i]
            whi[i] = wr[1,i]+vel/cspeed*wr[1,i]
            
        # For multi-order wavelengths, the final wavelength solution will extend
        # beyond the boundary of the original wavelength solution (at the end).
        # The extra pixels will have to be "padded" with masked out values.
        
        # We want the same logarithmic step for all order
        dw = np.log10(np.float64(wave[1:,:])) - np.log10(np.float64(wave[0:-1,:]))
        dwlog = np.median(np.abs(dw))   # positive

        # Extend at the ends
        if vel>0:
            nlo = np.empty(norder,int)
            nhi = np.empty(norder,int)
            for i in range(norder):
                nlo[i] = np.int(np.ceil((np.log10(np.float64(wr[0,i]))-np.log10(np.float64(wlo[i])))/dwlog))
                nhi[i] = np.int(np.ceil((np.log10(np.float64(whi[i]))-np.log10(np.float64(wr[1,i])))/dwlog))
            # Use the maximum over all orders
            nlo = np.max(nlo)
            nhi = np.max(nhi)
        else:
            nlo = 0
            nhi = 0

        # Number of pixels for the input wavelength range
        n = np.empty(norder,int)
        for i in range(norder):
            if vel==0:
                n[i] = np.int((np.log10(np.float64(wr[1,i]))-np.log10(np.float64(wr[0,i])))/dwlog)
            else:
                n[i] = np.int(np.ceil((np.log10(np.float64(wr[1,i]))-np.log10(np.float64(wr[0,i])))/dwlog))        
        # maximum over all orders
        n = np.max(n)

        # Final number of pixels
        nf = n+nlo+nhi

        # Final wavelength array
        fwave = np.empty((nf,norder),np.float64)
        for i in range(norder):
            fwave[:,i] = 10**( (np.arange(nf)-nlo)*dwlog+np.log10(np.float64(wr[0,i])) )

        # Make sure the element that's associated with the first input wavelength is identical
        for i in range(norder):
            # Increasing input wavelength arrays
            if np.median(dw[:,i])>0:
                fwave[:,i] -= fwave[nlo,i]-wave[0,i]
            # Decreasing input wavelength arrays
            else:
                fwave[:,i] -= fwave[nlo,i]-wave[-1,i]
        
    # Single-order wavelength array
    else:
        # extend wavelength range by +/-vel km/s
        wlo = wr[0]-vel/cspeed*wr[0]
        whi = wr[1]+vel/cspeed*wr[1]

        # logarithmic step
        dwlog = np.median(dln.slope(np.log10(np.float64(wave))))
        
        # extend at the ends
        if vel>0:
            nlo = np.int(np.ceil((np.log10(np.float64(wr[0]))-np.log10(np.float64(wlo)))/dwlog))    
            nhi = np.int(np.ceil((np.log10(np.float64(whi))-np.log10(np.float64(wr[1])))/dwlog))        
        else:
            nlo = 0
            nhi = 0

        # Number of pixels
        if vel==0.0:
            n = np.int((np.log10(np.float64(wr[1]))-np.log10(np.float64(wr[0])))/dwlog)
        else:
            n = np.int(np.ceil((np.log10(np.float64(wr[1]))-np.log10(np.float64(wr[0])))/dwlog))        
        nf = n+nlo+nhi

        fwave = 10**( (np.arange(nf)-nlo)*dwlog+np.log10(np.float64(wr[0])) )

        # Make sure the element that's associated with the first input wavelength is identical
        fwave -= fwave[nlo]-wave[0]
    
    # w=10**(w0log+i*dwlog)
    return fwave


# The doppler data directory
def datadir():
    """ Return the doppler data/ directory."""
    fil = os.path.abspath(__file__)
    codedir = os.path.dirname(fil)
    datadir = codedir+'/data/'
    return datadir

# Split a filename into directory, base and fits extensions
def splitfilename(filename):
    """ Split filename into directory, base and extensions."""
    fdir = os.path.dirname(filename)
    base = os.path.basename(filename)
    exten = ['.fit','.fits','.fit.gz','.fits.gz','.fit.fz','.fits.fz']
    for e in exten:
        if base[-len(e):]==e:
            base = base[0:-len(e)]
            ext = e
            break
    return (fdir,base,ext)

# Denoise a spectrum array
def denoise(flux,err=None,wtype='db2'):
    """ Try to remove the noise using wavelet denoising."""
    # sym9 seems to produce a smoother curve
    # db2 is also pretty good
    # harr and sb1 give more "blocky" results
    
    # Get shape
    if flux.ndim==1:
        npix = len(flux)
        norder = 1
    else:
        npix,norder = flux.shape

    # Loop over the orders
    out = np.zeros((npix,norder),float)
    for i in range(norder):
        if norder==1:
            f = flux
            e = err
        else:
            f = flux[:,0]
            e = err[:,0]

        # Normalize
        medf = np.median(f)
        f /= medf
        e /= medf
            
        # Values must be finite
        smlen = np.minimum(31,int(np.round(0.1*npix)))
        if smlen % 2 == 0:  # smlen must be odd
            smlen += 1
        bd,nbd = dln.where(~np.isfinite(f))
        if nbd>0:
            smf = dln.medfilt(f,smlen)
            f[bd] = smf[bd]
            sme = dln.medfilt(e,smlen)        
            e[bd] = sme[bd]

        # Get median err
        if err is not None:
            mede = np.median(e)
        else:
            mede = dln.mad(f-dln.medfilt(f,5))

        # Do the denoising, and rescale with medf
        out[:,i] = medf * denoise_wavelet(f, mede, wavelet=wtype, multichannel=False, method='BayesShrink', mode='soft', rescale_sigma=True)  

    # Only one order
    if norder==1:
        out = out.flatten()

    return out

#out=denoise_wavelet(f2,0.3, wavelet='db2',method='BayesShrink', mode='soft', rescale_sigma=False) 


def specprep(spec):
    """ Prepare the spectrum.  Mask bad pixels and normalize."""

    # Mask the spectrum
    if spec.mask is not None:
        # Set errors to high value, leave flux alone
        spec.err[spec.mask] = 1e30
    # Fix any NaNs in flux
    bd = np.where(~np.isfinite(spec.flux))
    if len(bd[0])>0:
        if spec.norder>1:
            spec.flux[bd[0],bd[1]] = 0.0
            spec.err[bd[0],bd[1]] = 1e30
            spec.mask[bd[0],bd[1]] = True
        else:
            spec.flux[bd[0]] = 0.0
            spec.err[bd[0]] = 1e30
            spec.mask[bd[0]] = True
    # normalize spectrum
    if spec.normalized is False: spec.normalize()  

    return spec


def maskoutliers(spec,nsig=5,verbose=False):
    """
    Mask large positive outliers and negative flux pixels in the spectrum.

    """

    spec2 = spec.copy()
    wave = spec2.wave.copy().reshape(spec2.npix,spec2.norder)   # make 2D
    flux = spec2.flux.copy().reshape(spec2.npix,spec2.norder)   # make 2D
    err = spec2.err.copy().reshape(spec2.npix,spec2.norder)     # make 2D
    mask = spec2.mask.copy().reshape(spec2.npix,spec2.norder)   # make 2D
    totnbd = 0
    for o in range(spec2.norder):
        w = wave[:,o].copy()
        x = (w-np.median(w))/(np.max(w*0.5)-np.min(w*0.5))  # -1 to +1
        y = flux[:,o].copy()
        m = mask[:,o].copy()
        # Divide by median
        medy = np.nanmedian(y)
        y /= medy
        # Perform sigma clipping out large positive outliers
        coef = dln.poly_fit(x,y,2,robust=True)
        sig = dln.mad(y-dln.poly(x,coef))
        bd,nbd = dln.where( ((y-dln.poly(x,coef)) > nsig*sig) | (y<0))
        totnbd += nbd
        if nbd>0:
            flux[bd,o] = dln.poly(x[bd],coef)*medy
            err[bd,o] = 1e30
            mask[bd,o] = True

    # Flatten to 1D if norder=1
    if spec2.norder==1:
        flux = flux.flatten()
        err = err.flatten()
        mask = mask.flatten()

    # Stuff back in
    spec2.flux = flux
    spec2.err = err
    spec2.mask = mask
    
    if verbose is True: print('Masked '+str(totnbd)+' outlier or negative pixels')
    
    return spec2


def maskdiscrepant(spec,model,nsig=10,verbose=False):
    """
    Mask pixels that are discrepant when compared to a model.

    """

    spec2 = spec.copy()
    wave = spec2.wave.copy().reshape(spec2.npix,spec2.norder)   # make 2D
    flux = spec2.flux.copy().reshape(spec2.npix,spec2.norder)   # make 2D
    err = spec2.err.copy().reshape(spec2.npix,spec2.norder)     # make 2D
    mask = spec2.mask.copy().reshape(spec2.npix,spec2.norder)   # make 2D
    mflux = model.flux.copy().reshape(spec2.npix,spec2.norder)   # make 2D
    totnbd = 0
    for o in range(spec2.norder):
        w = wave[:,o].copy()
        x = (w-np.median(w))/(np.max(w*0.5)-np.min(w*0.5))  # -1 to +1
        y = flux[:,o].copy()
        m = mask[:,o].copy()
        my = mflux[:,o].copy()
        # Divide by median
        medy = np.nanmedian(y)
        y /= medy
        my /= medy
        # Perform sigma clipping out large positive outliers
        coef = dln.poly_fit(x,y,2,robust=True)
        sig = dln.mad(y-my)
        bd,nbd = dln.where( np.abs(y-my) > nsig*sig )
        totnbd += nbd
        if nbd>0:
            flux[bd,o] = dln.poly(x[bd],coef)*medy
            err[bd,o] = 1e30
            mask[bd,o] = True

    # Flatten to 1D if norder=1
    if spec2.norder==1:
        flux = flux.flatten()
        err = err.flatten()
        mask = mask.flatten()

    # Stuff back in
    spec2.flux = flux
    spec2.err = err
    spec2.mask = mask

    if verbose is True: print('Masked '+str(totnbd)+' discrepant pixels')
    
    return spec2


def plotspec(spec,spec2=None,model=None,figsize=10):
    """ Plot a spectrum."""
    norder = spec.norder
    fig = plt.gcf()   # get current graphics window
    fig.clf()         # clear
    # Single-order plot
    if norder==1:
        ax = fig.subplots()
        fig.set_figheight(figsize*0.5)
        fig.set_figwidth(figsize)
        plt.plot(spec.wave,spec.flux,'b',label='Spectrum 1',linewidth=1)
        if spec2 is not None:
            plt.plot(spec2.wave,spec2.flux,'green',label='Spectrum 2',linewidth=1)            
        if model is not None:
            plt.plot(model.wave,model.flux,'r',label='Model',linewidth=1,alpha=0.8)
        leg = ax.legend(loc='upper left', frameon=False)
        plt.xlabel('Wavelength (Angstroms)')
        plt.ylabel('Normalized Flux')
        xr = dln.minmax(spec.wave)
        allflux = spec.flux.flatten()
        if spec2 is not None:
            allflux = np.concatenate((allflux,spec2.flux.flatten()))
        if model is not None:
            allflux = np.concatenate((allflux,model.flux.flatten()))                
        yr = [np.min(allflux), np.max(allflux)]
        yr = [yr[0]-dln.valrange(yr)*0.15,yr[1]+dln.valrange(yr)*0.005]
        yr = [np.max([yr[0],-0.2]), np.min([yr[1],2.0])]
        plt.xlim(xr)
        plt.ylim(yr)
    # Multi-order plot
    else:
        ax = fig.subplots(norder)
        fig.set_figheight(figsize)
        fig.set_figwidth(figsize)
        for i in range(norder):
            ax[i].plot(spec.wave[:,i],spec.flux[:,i],'b',label='Spectrum 1',linewidth=1)
            if spec2 is not None:
                ax[i].plot(spec2.wave[:,i],spec2.flux[:,i],'green',label='Spectrum 2',linewidth=1)                        
            if model is not None:
                ax[i].plot(model.wave[:,i],model.flux[:,i],'r',label='Model',linewidth=1,alpha=0.8)
            if i==0:
                leg = ax[i].legend(loc='upper left', frameon=False)
            ax[i].set_xlabel('Wavelength (Angstroms)')
            ax[i].set_ylabel('Normalized Flux')
            xr = dln.minmax(spec.wave[:,i])
            allflux = spec.flux[:,i].flatten()
            if spec2 is not None:
                allflux = np.concatenate((allflux,spec2.flux[:,i].flatten()))
            if model is not None:
                allflux = np.concatenate((allflux,model.flux[:,i].flatten()))                
            yr = [np.min(allflux), np.max(allflux)]
            yr = [yr[0]-dln.valrange(yr)*0.05,yr[1]+dln.valrange(yr)*0.05]
            if i==0:
                yr = [yr[0]-dln.valrange(yr)*0.15,yr[1]+dln.valrange(yr)*0.05]            
            yr = [np.max([yr[0],-0.2]), np.min([yr[1],2.0])]
            ax[i].set_xlim(xr)
            ax[i].set_ylim(yr)
