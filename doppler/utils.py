#!/usr/bin/env python

"""UTILS.PY - Utility functions

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20210605'  # yyyymmdd                                                                                                                           

import os
import time
import numpy as np
import warnings
import gdown
from scipy import sparse
from scipy.interpolate import interp1d
from dlnpyutils import utils as dln
import matplotlib.pyplot as plt
try:
    import __builtin__ as builtins # Python 2
except ImportError:
    import builtins # Python 3
        
# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

cspeed = 2.99792458e5  # speed of light in km/s

def getprintfunc(inplogger=None):
    """ Allows you to modify print() locally with a logger."""
    
    # Input logger
    if inplogger is not None:
        return inplogger.info  
    # Check if a global logger is defined
    elif hasattr(builtins,"logger"):
        return builtins.logger.info
    # Return the buildin print function
    else:
        return builtins.print
    

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

    Example
    -------
    .. code-block:: python

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
            #coef1 = dln.poly_fit(win[0:10], xin[0:10], 2)
            coef1 = dln.quadratic_coefficients(win[0:10], xin[0:10])            
            bd1, nbd1 = dln.where(w < np.min(dispersion))
            x[bd1] = dln.poly(w[bd1],coef1)
        # At the end
        if (np.max(w)>np.max(dispersion)):
            #coef2 = dln.poly_fit(win[npix-10:], xin[npix-10:], 2)
            coef2 = dln.quadratic_coefficients(win[npix-10:], xin[npix-10:])            
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

    Example
    -------
    .. code-block:: python

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
            #coef1 = dln.poly_fit(xin[0:10], win[0:10], 2)
            coef1 = dln.quadratic_coefficients(xin[0:10], win[0:10])            
            bd1, nbd1 = dln.where(x < 0)
            w[bd1] = dln.poly(x[bd1],coef1)
        # At the end
        if (np.max(x)>(npix-1)):
            #coef2 = dln.poly_fit(xin[npix-10:], win[npix-10:], 2)
            coef2 = dln.quadratic_coefficients(xin[npix-10:], win[npix-10:])            
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

    Example
    -------
    .. code-block:: python

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

    Example
    -------
    .. code-block:: python

         out = convolve_sparse(spec,lsf)

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

def gausskernel(wave,vgauss):
    """
    Creates a 2D Gaussian kernel for an input wavelength array.

    Parameters
    ----------
    wave : numpy array
       Input 1D wavelength array for which to create the Gaussian kernel.
    vgauss : float
       The input Gaussian sigma velocity in km/s.

    Returns
    -------
    kernel : numpy array
       The output 2D Gaussian kernel.

    Example
    -------
    .. code-block:: python

         kernel = gausskernel(wave,10.0)

    """
    # Create a 2D Gaussian broadening kernel array
    npix = len(np.atleast_1d(wave))
    dw = np.diff(wave)
    dw = np.hstack((dw,dw[-1]))
    dw = np.abs(dw)

    # Vgauss is the velocity FWHM
    gsigma = np.float64(vgauss*wave/(cspeed*dw*2.3548))
    
    # How many pixels do we need to capture the profile
    minhalfpix = gsigma*3
    if np.max(minhalfpix) < 0.1:
        return np.ones(npix).astype(np.float64)
    else:
        nkernel = np.int(np.max(np.ceil(minhalfpix)))*2 + 1
    
    xkernel = np.arange(nkernel)-nkernel//2
    xkernel2 = np.repeat(xkernel,npix).reshape((nkernel,npix)).T
    gsigma2 = np.repeat(gsigma,nkernel).reshape((npix,nkernel))
    kernel = np.exp(-0.5*xkernel2**2 / gsigma2**2) / (np.sqrt(2*np.pi)*gsigma2)
    kernel[kernel<0.] = 0.
    kernel /= np.tile(np.sum(kernel,axis=1),(nkernel,1)).T

    return kernel


def rotkernel(wave,vsini,eps=0.6):
    """
    Creates a 2D rotational kernel for an input wavelength array.

    Parameters
    ----------
    wave : numpy array
       Input 1D wavelength array for which to create the rotational kernel.
    vsini : float
       The rotational velocity in km/s.
    eps : float, optional
       The linear limb-darkening coefficient (epsilon, 0-1) to use for the
       rotational profile.  Default is 0.6.

    Returns
    -------
    kernel : numpy array
       The output 2D rotational kernel.

    Example
    -------
    .. code-block:: python

         kernel = rotkernel(wave,10.0)

    """

    # Create a 2D Rotational broadening kernel array

    # G(dlambda) = 2*(1-eps)*sqrt(1-(dlambda/dlambdaL)^2) + 0.5*pi*eps*(1-(dlambda/dlambdaL)^2)
    #              ----------------------------------------------------------------------------
    #                                  pi*dlambdaL*(1-epsilon/3)
    # dlambda = offset from line center (A)
    # dlambdaL = vsini in wavelength units (A)
    
    npix = len(np.atleast_1d(wave))
    dw = np.diff(wave)
    dw = np.hstack((dw,dw[-1]))
    dw = np.abs(dw)
    
    # How many pixels do we need to capture the rotational profile
    minhalfpix = np.ceil(wave*vsini/(cspeed*dw))
    nkernel = np.int(np.max( 2*minhalfpix+1 ))
    
    # Vsini in units of pixels
    xvsini = wave*vsini/(cspeed*dw)
    xvsini2 = np.repeat(xvsini,nkernel).reshape((npix,nkernel))

    xkernel = np.arange(nkernel)-nkernel//2    
    xkernel2 = np.repeat(xkernel,npix).reshape((nkernel,npix)).T

    vfrac = xkernel2/xvsini2
    kernel = np.where(np.abs(vfrac) <= 1,
                      2*(1-eps)*np.sqrt(1-vfrac**2) + 0.5*np.pi*eps*(1-vfrac**2), 0)
    kernel /= np.pi*xvsini2*(1-eps/3.0)

    kernel[kernel<0.] = 0.
    kernel /= np.tile(np.sum(kernel,axis=1),(nkernel,1)).T
    
    return kernel


def check_rotkernel(wave,vsini,eps=0.6):
    """ Helper function to check if the rotational kernel will have any effect."""
    # it will return True if it will have an effect and False if not
    kernel = rotkernel(wave,vsini,eps=eps)
    if kernel.shape[1]==3 and np.sum(kernel[:,1]>0.99)==kernel.shape[0]:
        return False
    return True
    

#def combine_kernels(kernels):
#    # Combine mulitple 2D kernels, convolve them with each other
#    # input list or tuple of kernels
#
#    n = len(kernels)
#    npix = kernels[0].shape[0]
#    nkernelarr = np.zeros(n,int)
#    for i,k in enumerate(kernels):
#        ndim = k.ndim
#        if ndim>1: nkernelarr[i]=k.shape[1]
#
#    nkernel = np.int(np.max(nkernelarr))
#    kernel = np.zeros((npix,nkernel),np.float64)
#    for i,k in enumerate(kernels):
#        if nkernelarr[i]>1:
#            kernel1 = np.zeros((npix,nkernel),np.float64)            
#            off = (nkernel-nkernelarr[i])//2
#            kernel1[:,off:off+nkernelarr[i]] = k
#            # Convolve them with each other
#            kernel += kernel1**2    # add in quadrature
#    # Take sqrt() or quadrature
#    kernel = np.sqrt(kernel)
#    # Normalize
#    # Make sure it's normalized
#    kernel /= np.tile(np.sum(kernel,axis=1),(nkernel,1)).T
#
#    return kernel

# Rotational and Gaussian broaden a spectrum
def broaden(wave,flux,vgauss=None,vsini=None):
    """
    Broaden a spectrum with Gaussian and Rotational broadening.

    Parameters
    ----------
    wave : numpy array
       Wavelength array of the spectrum.
    flux : numpy array
       Flux array of the spectrum.
    vgauss : float
       The input Gaussian sigma velocity in km/s. Optional.
    vsini : float
       The rotational velocity in km/s.  Optional.
    eps : float, optional
       The linear limb-darkening coefficient (epsilon, 0-1) to use for the
       rotational profile.  Default is 0.6.

    Returns
    -------
    oflux : numpy array
       The output broadened flux array.

    Example
    -------
    .. code-block:: python

         oflux = broaden(wave,flux,vgauss=10.0,vsini=20.0)

    """
    
    # Create the 2D broadening kernel array
    npix = len(wave)
    # Are the wavelengths logarithmically spaced
    logwave = False
    if dln.valrange(np.diff(np.log(wave))) < 1e-7: logwave=True

    # If the wavelengths are on a logarithmic wavelength scale
    #  then just convolve with a 1D kernel with np.convolve(flux,kernel,mode='same')


    # Check the rotational kernel to see if it will have any effect
    #  this can happen if vsini is low (<~ 3.5 km/s).
    # If it has no effect, then use Vgauss instead
    #  note that this will also have a small (but nonzero) effect
    if vsini is not None:
        if vsini>0.0:
            if check_rotkernel(wave,vsini) is False:
                if vgauss is None:
                    vgauss = vsini
                else:
                    vgauss = np.sqrt(vgauss**2+vsini**2)  # add in quadrature
                vsini = 0.0
                
    # Initializing output flux
    oflux = flux.copy()
    
    # Add Gaussian broadening
    if vgauss is not None:
        if vgauss != 0.0:
            # 2D kernel
            if logwave==False:
                gkernel = gausskernel(wave,vgauss)
                # Don't use this if it's not going to do anything!!!
                if gkernel.ndim==2:
                    oflux = convolve_sparse(oflux,gkernel)
            # 1D kernel, log wavelengths
            else:
                gkernel = gausskernel(wave[0:2],vgauss)
                if (gkernel.ndim==1) and (gkernel[0]>0.99):  # nothing happened
                    return flux
                oflux = np.convolve(oflux,gkernel[0,:],mode='same')
            
    # Add Rotational broadening
    if vsini is not None:
        if vsini != 0.0:
            # 2D kernel
            if logwave==False:
                rkernel = rotkernel(wave,vsini)
                # Don't use this if it's not going to do anything!!!
                if rkernel.ndim==2:
                    oflux = convolve_sparse(oflux,rkernel)
            # 1D kernel, log wavelengths
            else:
                rkernel = rotkernel(wave[0:2],vsini)
                if (rkernel.ndim==1) and (rkernel[0]>0.99):  # nothing happened
                    return flux                
                oflux = np.convolve(oflux,rkernel[0,:],mode='same')

    return oflux


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

    Example
    -------
    .. code-block:: python

         fwave = make_logwave_scale(wave)

    """

    # If the existing wavelength scale is logarithmic then use it, just extend
    # on either side
    vel = np.abs(vel)
    wave = np.float64(wave)
    nw = len(wave)
    wr = dln.minmax(wave)

    # Multi-order wavelength array
    if wave.ndim==2:
        npix,norder = wave.shape

        # Ranges
        wr = np.zeros((2,norder),np.float64)
        wlo = np.zeros(norder,np.float64)
        whi = np.zeros(norder,np.float64)
        dw = np.zeros((npix,norder),np.float64)
        for i in range(norder):
            gdwave, = np.where((wave[:,i]>0) & np.isfinite(wave[:,i]))
            wr[:,i] = dln.minmax(wave[gdwave,i])
            dw[gdwave,i] = np.gradient(np.log10(np.float64(wave[gdwave,i])))
            wlo[i] = wr[0,i]-vel/cspeed*wr[0,i]
            whi[i] = wr[1,i]+vel/cspeed*wr[1,i]
            
        # For multi-order wavelengths, the final wavelength solution will extend
        # beyond the boundary of the original wavelength solution (at the end).
        # The extra pixels will have to be "padded" with masked out values.
        
        # We want the same logarithmic step for all order
        #dw = np.log10(np.float64(wave[1:,:])) - np.log10(np.float64(wave[0:-1,:]))
        dwlog = np.median(np.abs(dw))   # positive
        
        # Extend at the ends
        if vel>0:
            nlo = np.zeros(norder,int)
            nhi = np.zeros(norder,int)
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
        n = np.zeros(norder,int)
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
        fwave = np.zeros((nf,norder),np.float64)
        for i in range(norder):
            fwave[:,i] = 10**( (np.arange(nf)-nlo)*dwlog+np.log10(np.float64(wr[0,i])) )

        # Make sure the element that's associated with the first input wavelength is identical
        for i in range(norder):
            gdwave, = np.where((wave[:,i]>0) & np.isfinite(wave[:,i]))
            # Increasing input wavelength arrays
            if np.median(dw[gdwave,i])>0:
                #fwave[:,i] -= fwave[nlo,i]-wave[0,i]
                fwave[:,i] -= fwave[nlo,i]-wave[gdwave[0],i]
            # Decreasing input wavelength arrays
            else:
                #fwave[:,i] -= fwave[nlo,i]-wave[-1,i]
                fwave[:,i] -= fwave[nlo,i]-wave[gdwave[-1],i]
        
    # Single-order wavelength array
    else:
        # extend wavelength range by +/-vel km/s
        wlo = wr[0]-vel/cspeed*wr[0]
        whi = wr[1]+vel/cspeed*wr[1]

        # logarithmic step
        gdwave, = np.where((wave>0) & np.isfinite(wave))
        dwlog = np.median(np.gradient(np.log10(np.float64(wave[gdwave]))))
        
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

    # Check for orders that are entirely masked
    if spec.norder>1:
        ngood = np.sum(~spec.mask,axis=0)
        bdorder, = np.where(ngood==0)
        if len(bdorder)==spec.norder:
            raise ValueError('Entire spectrum is masked')
        if len(bdorder)>0:
            print('Removing order '+str(bdorder)+' that are entirely masked')
            np.delete(spec.flux,bdorder,axis=1)
            np.delete(spec.err,bdorder,axis=1)
            np.delete(spec.wave,bdorder,axis=1)
            np.delete(spec.mask,bdorder,axis=1)
            np.delete(spec.lsf.wave,bdorder,axis=1)
            np.delete(spec.lsf.pars,bdorder,axis=1)
            spec.norder -= 1
    else:
        ngood = np.sum(~spec.mask)
        if ngood==0:
            raise ValueError('Entire spectrum is masked')
            
    # normalize spectrum
    if spec.normalized is False: spec.normalize()  

    return spec


def maskoutliers(spec,nsig=5,verbose=False):
    """
    Mask large positive outliers and negative flux pixels in the spectrum.

    Parameters
    ----------
    spec : Spec1D object
       Observed spectrum to mask outliers on.
    nsig : int, optional
       Number of standard deviations to use for the outlier rejection.
       Default is 5.0.
    verbose : boolean, optional
       Verbose output.  Default is False.

    Returns
    -------
    spec2 : Spec1D object
       Spectrum with outliers masked.

    Example
    -------
    .. code-block:: python

         spec = maskoutliers(spec,nsig=5)

    """

    print = getprintfunc() # Get print function to be used locally, allows for easy logging
    
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
        gpix, = np.where(~m & np.isfinite(y))
        if len(gpix)==0:
            continue
        # Divide by median
        medy = np.nanmedian(y[gpix])
        if medy <= 0.0:
            medy = 1.0
        y[gpix] /= medy
        # Perform sigma clipping out large positive outliers
        coef = dln.poly_fit(x[gpix],y[gpix],2,robust=True)
        sig = dln.mad(y[gpix]-dln.poly(x[gpix],coef))
        bd1,nbd = dln.where( ((y[gpix]-dln.poly(x[gpix],coef)) > nsig*sig) | (y[gpix]<0))
        if nbd>0:
            bd = gpix[bd1]
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

    Parameters
    ----------
    spec : Spec1D object
       Observed spectrum for which to mask discrepant values.
    model : Spec1D object
       Reference/model spectrum to use to find discrepant values.
    nsig : int, optional
       Number of standard deviations to use for the discrepant values.
       Default is 10.0.
    verbose : boolean, optional
       Verbose output.  Default is False.

    Returns
    -------
    spec2 : Spec1D object
       Spectrum with discrepant values masked.

    Example
    -------
    .. code-block:: python

         spec = maskdiscrepant(spec,nsig=5)

    """

    print = getprintfunc() # Get print function to be used locally, allows for easy logging
    
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
        if medy <= 0.0:
            medy = 1.0
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

def download_data(force=False):
    """ Download the data from my Google Drive."""

    # Check if the "done" file is there
    if os.path.exists(datadir()+'done') and force==False:
        return
    
    #https://drive.google.com/drive/folders/1SXId9S9sduor3xUz9Ukfp71E-BhGeGmn?usp=share_link
    # The entire folder: 1SXId9S9sduor3xUz9Ukfp71E-BhGeGmn
    
    data = [{'id':'17EDOUbzNr4cDzn7KZdPsw7R2lAj_o_np','output':'cannongrid_3000_18000_hotdwarfs_norm_cubic_model.pkl'},   
            {'id':'1hy1Mk6Kl8OAH6UouyLeGwtBXsOKoDq1r','output':'cannongrid_3000_18000_coolstars_norm_cubic_model.pkl'},
            {'id':'16FP-udWusOSuIghpZGmQagvYwnFlS33C','output':'payne_coolhot_0.npz'},
            {'id':'1zgFM1UhigHvbQ_oUj_k_S99APS39nwRp','output':'payne_coolhot_1.npz'},
            {'id':'1bGTBO8LdEpbhGztP-Rj5xvAMpocCINeP','output':'payne_coolhot_2.npz'},
            {'id':'1iCDKTXcS1JK75in8hBnzdevW1VFYU8od','output':'payne_coolhot_3.npz'},
            {'id':'13gsGHalkJCeOFoE0edLKVzMXPG5ZRhWA','output':'payne_coolhot_4.npz'},
            {'id':'16hsj29H2gzZIfdSxdbvp6eWqtZ7wn3FP','output':'payne_coolhot_5.npz'},
            {'id':'1slMxAg8P9wjomr0wVyHYmpN1SkL4dtCa','output':'payne_coolhot_6.npz'},
            {'id':'13-S4Ecr9_6VJ8TTQoSgYCeCSF4ntfX0M','output':'payne_coolhot_7.npz'},
            {'id':'1-9B1Netb5I1rDSyo6l-iQup0m_mAA0UV','output':'payne_coolhot_8.npz'},            
            {'id':'1cMM0AJ49hvhZdeJpKoGAnlwcDep7GNcX','output':'payne_coolhot_9.npz'},
            {'id':'1uYUyVuJg-YVqx2zczunUYZTnbfg-tbdL','output':'payne_coolhot_10.npz'},
            {'id':'1u6HlTfSJ9D_sabbnbPKce0oBZ50n3YJj','output':'payne_coolhot_11.npz'},
            {'id':'14hnD2CLWAE0WdpByhIjDkDQWOZRfGEkn','output':'payne_coolhot_12.npz'},
            {'id':'17M1Zp55L1eBlzDDvKvxE9xk998K1m7v5','output':'payne_coolhot_13.npz'},
            {'id':'1uFArsToJyS2tMJLt7PSvOQMHbwXcT4rt','output':'payne_coolhot_14.npz'},
            {'id':'1q1tG_0E8907PrwG88idVzFYjO3UmOtW_','output':'payne_coolhot_15.npz'},
            {'id':'1zBCHtlLHpvUkyocqYBlbaeH8IvzEsxCz','output':'payne_coolhot_16.npz'},
            {'id':'1_Ab3ArNekoNUyGoQ_eGlbLEE_p6IxNTf','output':'payne_coolhot_17.npz'},
            {'id':'1UQvZ5KDe1tHcyV9jzQnAa-YIyV5RDPmp','output':'payne_coolhot_18.npz'},
            {'id':'1JIVkXtedeeZVQjsewDJdzdZRIcJhmQWm','output':'payne_coolhot_19.npz'},
            {'id':'1VpDd2KO0sZMntCvLjzcv6por2vKs-R3D','output':'payne_coolhot_20.npz'},
            {'id':'1NXG0g3yaNbcgPv0wV4awsonFT78Kmzka','output':'payne_coolhot_21.npz'},
            {'id':'1u130026I7qvG9w1GQFymPiwP9kGkuicP','output':'payne_coolhot_22.npz'},
            {'id':'1-MYvm-aXGsH4kZ_Hi1b_9YFI0y0IqJg2','output':'payne_coolhot_23.npz'},
            {'id':'1VTWHL2Ie4wFSQSsKz9oowYRK4om0vsB8','output':'payne_coolhot_24.npz'},
            {'id':'1XzIyErtLAI89aB5xj2zDgBMB3MZUMMgb','output':'payne_coolhot_25.npz'},
            {'id':'12rqcJIpjUEJlP11onfbCyxy40nrsA-S8','output':'payne_coolhot_26.npz'},
            {'id':'1wp1UCkFGXvWiEtFiNYRutPAemwXX67ad','output':'payne_coolhot_27.npz'},
            {'id':'1j9sBofKrAI_Mgocq1MdxftlnHnakaWUo','output':'payne_coolhot_28.npz'},
            {'id':'1w4CwoZsxEyBRs7DvrZ4P68PMBQuhAjav','output':'payne_coolhot_29.npz'}]
    
    # This should take 2-3 minutes on a good connection
    
    # Do the downloading
    t0 = time.time()
    print('Downloading '+str(len(data))+' Doppler data files')
    for i in range(len(data)):
        print(str(i+1)+' '+data[i]['output'])
        fileid = data[i]['id']
        url = f'https://drive.google.com/uc?id={fileid}'
        output = datadir()+data[i]['output']  # save to the data directory
        if os.path.exists(output)==False or force:
            gdown.download(url, output, quiet=False)

    print('All done in {:.1f} seconds'.format(time.time()-t0))

            

