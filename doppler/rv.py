#!/usr/bin/env python

"""RV.PY - Generic Radial Velocity Software

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20210605'  # yyyymmdd                                                                                                                           

import os
#import sys, traceback
import contextlib, io, sys
import numpy as np
import warnings
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.wcs import WCS
from scipy.ndimage.filters import median_filter,gaussian_filter1d
from dlnpyutils.minpack import curve_fit
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
import thecannon as tc
from dlnpyutils import utils as dln, bindata
from .spec1d import Spec1D
from . import (cannon,payne,utils,reader)
import copy
import emcee
import corner
import logging
import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
import subprocess
try:
    import __builtin__ as builtins # Python 2
except ImportError:
    import builtins # Python 3
    
# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

cspeed = 2.99792458e5  # speed of light in km/s

def xcorr_dtype(nlag):
    """Return the dtype for the xcorr structure"""
    dtype = np.dtype([("xshift0",float),("ccp0",float),("xshift",float),("xshifterr",float),
                      ("xshift_interp",float),("ccf",(float,nlag)),("ccferr",(float,nlag)),("ccnlag",int),
                      ("cclag",(int,nlag)),("ccpeak",float),("ccpfwhm",float),("ccp_pars",(float,4)),
                      ("ccp_perror",(float,4)),("ccp_polycoef",(float,4)),("vrel",float),
                      ("vrelerr",float),("w0",float),("dw",float),("chisq",float)])
    return dtype

# astropy.modeling can handle errors and constraints


# https://stackoverflow.com/questions/2828953/silence-the-stdout-of-a-function-in-python-without-trashing-sys-stdout-and-resto
# usage:
#  with mute():
#    foo()
@contextlib.contextmanager
def mute():
    '''Prevent print to stdout, but if there was an error then catch it and
    print the output before raising the error.'''

    saved_stdout = sys.stdout
    saved_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()    
    try:
        yield
    except Exception:
        saved_output = sys.stdout
        saved_outerr = sys.stderr
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr
        print(saved_output.getvalue())
        print(saved_outerr.getvalue())        
        raise
    sys.stdout = saved_stdout
    sys.stderr = saved_stderr

    
def tweakcontinuum(spec,model,smlen=None,usepoly=False,polyorder=3):
    """ Tweak the continuum normalization of an observed spectrum using a good-fit model."""

    if smlen is None:
        smlen = spec.npix/10.0
        
    if hasattr(spec,'cont') is False:
        spec.cont = spec.flux.copy()*0+1

    # Loop over orders
    for i in range(spec.norder):
        if spec.norder==1:
            ratio = spec.flux/model.flux
            mask = spec.mask
            gd,ngd,bd,nbd = dln.where((mask==False) & (spec.flux>0) & np.isfinite(spec.flux),comp=True)            
        else:
            ratio = spec.flux[:,i]/model.flux[:,i]
            mask = spec.mask[:,i]
            gd,ngd,bd,nbd = dln.where((mask==False) & (spec.flux[:,i]>0) & np.isfinite(spec.flux[:,i]),comp=True)
        # Set bad pixels to NaN, gsmooth masks those out
        if nbd>0:
            ratio[bd] = np.nan
        ratio[0] = np.nanmedian(ratio[0:np.int(smlen/2)])
        ratio[-1] = np.nanmedian(ratio[-np.int(smlen/2):-1])
        # Use Gaussian Smoothing
        if usepoly is False:
            sm = dln.gsmooth(ratio,smlen,boundary='extend')
        # Use low-order,polynomial
        else:
            x = np.linspace(-1,1,spec.npix)
            coef = dln.poly_fit(x[gd],ratio[gd]-1,polyorder)
            sm = dln.poly(x,coef)+1
        # Deal with any remaining NaNs
        bd, = np.where(np.isfinite(sm)==False)
        if len(bd)>0:
            sm[bd] = 1.0
        if spec.norder==1:
            spec.cont *= sm
            spec.flux /= sm
            spec.err /= sm
        else:
            spec.cont[:,i] *= sm
            spec.flux[:,i] /= sm
            spec.err[:,i] /= sm
            
    return spec


def specfigure(figfile,spec,fmodel,out,annotlabels=None,original=None,verbose=True,figsize=10,ispayne=False):
    """
    Make diagnostic figure.

    Parameters
    ----------
    figfile : str
       Output figure filename.
    spec : Spec1D object
       The observed spectrum used for the fitting (normalized, masked and tweaked).
    fmodel : Spec1D object
       The best-fitting model (Cannon or Payne) spectrum.
    out : numpy structured array
       Catalog of best-fitting values to use for the annotation.
    annotlabels : list, optional
       The list of labels to use for the annotation.  Default is ['teff','logg','feh','vrel'].
    original : Spec1D object, optional
       The original but normalized spectrum.  Not tweaked or masked.
    verbose : boolean, optional
       Verbose output.  Default is True.
    figsize : float, optional
       Figure size to use.  Default is 10 inches.
    ispayne: boolean, optional
       Is this a Payne model.  Default is False.

    Returns
    -------
    Figure is saved to figfile.

    Example
    -------
    .. code-block:: python
           
         specfigure(figfile,specm,fmodel,out,original=orig,verbose=True,ispayne=True,annotlabels=annotlabels)

    """

    print = utils.getprintfunc() # Get print function to be used locally, allows for easy logging    
    
    if annotlabels is None:
        annotlabels = ['teff','logg','feh','vrel']
    #import matplotlib
    backend = matplotlib.rcParams['backend']
    matplotlib.use('Agg')
    #import matplotlib.pyplot as plt
    if os.path.exists(figfile): os.remove(figfile)
    norder = spec.norder
    nlegcol = 2
    if original is not None: nlegcol=3
    # Single-order plot
    if norder==1:
        fig,ax = plt.subplots()
        fig.set_figheight(figsize*0.5)
        fig.set_figwidth(figsize)
        if original is not None:
            plt.plot(original.wave,original.flux,color='green',label='Original',linewidth=1)
        plt.plot(spec.wave,spec.flux,'b',label='Masked Data',linewidth=1)
        plt.plot(fmodel.wave,fmodel.flux,'r',label='Model',linewidth=1,alpha=0.8)
        leg = ax.legend(loc='upper left', frameon=True, framealpha=0.8, ncol=nlegcol)
        plt.xlabel('Wavelength (Angstroms)')
        plt.ylabel('Normalized Flux')
        xr = dln.minmax(spec.wave)
        yr = [np.min([spec.flux,fmodel.flux]), np.max([spec.flux,fmodel.flux])]
        if original is not None:
            yr = [np.min([original.flux,spec.flux,fmodel.flux]), np.max([spec.flux,fmodel.flux])]            
        yr = [yr[0]-dln.valrange(yr)*0.15,yr[1]+dln.valrange(yr)*0.15]
        yr = [np.max([yr[0],-0.2]), np.min([yr[1],2.0])]
        plt.xlim(xr)
        plt.ylim(yr)
        snr = np.nanmedian(spec.flux/spec.err)
        plt.title(spec.filename)
        if ispayne is False:
            ax.annotate(r'S/N=%5.1f   Teff=%5.1f$\pm$%5.1f  logg=%5.2f$\pm$%5.2f  [Fe/H]=%5.2f$\pm$%5.2f   Vrel=%5.2f$\pm$%5.2f   chisq=%5.2f' %
                        (snr, out['teff'], out['tefferr'], out['logg'], out['loggerr'], out['feh'], out['feherr'], out['vrel'], out['vrelerr'], out['chisq']),
                        xy=(np.mean(xr), yr[0]+dln.valrange(yr)*0.05),ha='center')
        else:
            fmt = r'S/N=%5.1f '
            annotarr = [snr]
            for k,name in enumerate(annotlabels):
                fmt += r''+name+'=%5.1f$\pm$%5.1f  '
                annotarr += [dln.first_el(out[name]),dln.first_el(out[name+'err'])]
            fmt += r' chisq=%5.2f'
            annotarr += [dln.first_el(out['chisq'])]
            ax.annotate(fmt % tuple(annotarr), xy=(np.mean(xr), yr[0]+dln.valrange(yr)*0.05),ha='center')
            
    # Multi-order plot
    else:
        fig,ax = plt.subplots(norder)
        fig.set_figheight(figsize)
        fig.set_figwidth(figsize)
        for i in range(norder):
            if original is not None:
                ax[i].plot(original.wave[:,i],original.flux[:,i],color='green',label='Original',linewidth=1)            
            ax[i].plot(spec.wave[:,i],spec.flux[:,i],'b',label='Masked Data',linewidth=1)
            ax[i].plot(fmodel.wave[:,i],fmodel.flux[:,i],'r',label='Model',linewidth=1,alpha=0.8)
            if i==0:
                leg = ax[i].legend(loc='upper left', frameon=True, framealpha=0.8, ncol=nlegcol)
            ax[i].set_xlabel('Wavelength (Angstroms)')
            ax[i].set_ylabel('Normalized Flux')
            xr = dln.minmax(spec.wave[:,i])
            yr = [np.min([spec.flux[:,i],fmodel.flux[:,i]]), np.max([spec.flux[:,i],fmodel.flux[:,i]])]
            if original is not None:
                yr = [np.min([original.flux[:,i],spec.flux[:,i],fmodel.flux[:,i]]), np.max([spec.flux[:,i],fmodel.flux[:,i]])]
            yr = [yr[0]-dln.valrange(yr)*0.05,yr[1]+dln.valrange(yr)*0.05]
            if i==0:
                yr = [yr[0]-dln.valrange(yr)*0.15,yr[1]+dln.valrange(yr)*0.05]            
            yr = [np.max([yr[0],-0.2]), np.min([yr[1],2.0])]
            ax[i].set_xlim(xr)
            ax[i].set_ylim(yr)
            # legend
            if i==0:
                snr = np.nanmedian(spec.flux/spec.err)
                ax[i].set_title(spec.filename)
                if ispayne is False:
                    ax[i].annotate(r'S/N=%5.1f   Teff=%5.1f$\pm$%5.1f  logg=%5.2f$\pm$%5.2f  [Fe/H]=%5.2f$\pm$%5.2f   Vrel=%5.2f$\pm$%5.2f   chisq=%5.2f' %
                                   (snr,out['teff'],out['tefferr'],out['logg'],out['loggerr'],out['feh'],out['feherr'],out['vrel'],out['vrelerr'],out['chisq']),
                                   xy=(np.mean(xr), yr[0]+dln.valrange(yr)*0.05),ha='center')
                else:
                    fmt = r'S/N=%5.1f '
                    annotarr = [snr]
                    for k,name in enumerate(annotlabels):
                        fmt += r''+name+'=%5.1f$\pm$%5.1f  '
                        annotarr += [dln.first_el(out[name]),dln.first_el(out[name+'err'])]
                    fmt += r' chisq=%5.2f'
                    annotarr += [dln.first_el(out['chisq'])]
                    ax[i].annotate(fmt % tuple(annotarr), xy=(np.mean(xr), yr[0]+dln.valrange(yr)*0.05),ha='center')
    plt.savefig(figfile,bbox_inches='tight')
    plt.close(fig)
    if verbose is True: print('Figure saved to '+figfile)
    matplotlib.use(backend)  # back to the original backend

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
      The first array to cross correlate (e.g., the template).  If y is 2D
      (e.g., [Npix,Norder]), then x can be either 2D or 1D.  If x is 1D, then
      the cross-correlation is performed between x and each order of y and
      a 2D array will be output.  If x is 2D, then the cross-correlation
      of each order of x and y is performed and the results combined.
    y : array
      The second array to cross correlate.  Must be the same lenght as x.
      Can be 2D (e.g., [Npix, Norder]), but the shifting is always done
      on the 1st dimension.
    lag : array
      Vector that specifies the absolute distance(s) between
             indexed elements of x in the interval [-(n-2), (n-2)].
    yerr : array, optional
       Array of uncertainties in y.  Must be the same shape as y.
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

    xshape = x.shape
    yshape = y.shape
    nx = xshape[0]
    if x.ndim==1:
        nxorder = 1
    else:
        nxorder = xshape[1]
    ny = yshape[0]
    if y.ndim==1:
        nyorder = 1
    else:
        nyorder = yshape[1]
    npix = nx
    norder = np.maximum(nxorder,nyorder)

    # Check the inputs
    if (nx != len(y)):
        raise ValueError("X and Y arrays must have the same number of pixels in 1st dimension.")

    if (x.ndim>2) | (y.ndim>2):
        raise ValueError("X and Y must be 1D or 2D.")

    if (x.ndim==2) & (y.ndim==1):
        raise ValueError("If X is 2D then Y must be as well.")

    # If X and Y are 2D then their Norders must be the same
    if (x.ndim==2) & (y.ndim==2) & (nxorder!=nyorder):
        raise ValueError("If X and Y are 2D then their length in the 2nd dimension must be the same.")

    # Check that Y and Yerr have the same length
    if yerr is not None:
        if (y.shape != yerr.shape):
            raise ValueError("Y and Yerr must have the same shape.")
    
    if (nx<2):
        raise ValueError("X and Y arrays must contain 2 or more elements.")
    
    # Reshape arrays to [Npix,Norder], even if both are 1D
    xd = x.copy()
    yd = y.copy()
    if yerr is not None: yderr=yerr.copy()
    if (norder>1):
        if (x.ndim==1):
            # make multiple copies of X
            xd = yd.copy()*0.0
            for i in range(norder):
                xd[:,i] = x.copy()
    else:
        xd = xd.reshape(npix,1)
        yd = yd.reshape(npix,1)        
        if yerr is not None:
            yderr = yderr.reshape(npix,1)
        
    # Remove the means
    if nomean is False:
        for i in range(norder):
            xd[:,i] -= np.nanmean(xd[:,i])
            yd[:,i] -= np.nanmean(yd[:,i])
      
    # Set NaNs or Infs to 0.0, mask bad pixels
    fx = np.isfinite(xd)
    ngdx = np.sum(fx)
    nbdx = np.sum((fx==False))
    if nbdx>0: xd[(fx==False)]=0.0
    fy = np.isfinite(yd)
    if yerr is not None:
        fy &= (yderr<1e20)   # mask out high errors as well
    ngdy = np.sum(fy)
    nbdy = np.sum((fy==False))
    if nbdy>0:
        yd[(fy==False)]=0.0
        if yerr is not None: yderr[(fy==False)]=0.0
    nlag = len(lag)

    # MULTI-ORDER with ONE template and ONE spectrum to cross-correlate
    ###################################################################
    # This is 2.3x faster than the original method down below
    #   for (4936, 3) spectra and nlag=765
    if (norder>1) & (nxorder == nyorder):

        # Initialize the output arrays
        cross = np.zeros(nlag,dtype=float)
        cross_error = np.zeros(nlag,dtype=float)
        num = np.zeros(nlag,dtype=int)  # number of "good" points at this lag        
        
        # flatten multi-order to 1D with buffer
        buff = np.max(np.abs(lag))
        xd1 = np.zeros(nx*norder + (norder-1)*buff,float)
        yd1 = np.zeros(nx*norder + (norder-1)*buff,float)
        yderr1 = np.zeros(nx*norder + (norder-1)*buff,float)
        fx1 = np.zeros(nx*norder + (norder-1)*buff,bool)
        fy1 = np.zeros(nx*norder + (norder-1)*buff,bool)    
        for i in range(norder):
            lo = i*(nx+buff)
            hi = lo+nx
            xd1[lo:hi] = xd[:,i]
            yd1[lo:hi] = yd[:,i]
            yderr1[lo:hi] = yderr[:,i]
            fx1[lo:hi] = fx[:,i]
            fy1[lo:hi] = fy[:,i]        
        nx1 = len(xd1)
            
        # Loop over lag points
        for k in range(nlag):
            # Note the reversal of the variables for negative lags.
            if lag[k]>0:
                cross[k] = np.sum(xd1[0:nx1-lag[k]] * yd1[lag[k]:])
                num[k] = np.sum(fx1[0:nx1-lag[k]] * fy1[lag[k]:]) 
                if yerr is not None:
                    cross_error[k] = np.sum( (xd1[0:nx1-lag[k]] * yderr1[lag[k]:])**2 )
            else:
                cross[k] =  np.sum(yd1[0:nx1+lag[k]] * xd1[-lag[k]:])
                num[k] = np.sum(fy1[0:nx1+lag[k]] * fx1[-lag[k]:])
                if yerr is not None:
                    cross_error[k] = np.sum( (yderr1[0:nx1+lag[k]] * xd1[-lag[k]:])**2 )

        rmsx = np.sqrt(np.sum((xd*fx)**2))
        if rmsx==0.0: rmsx=1.0
        rmsy = np.sqrt(np.sum((yd*fy)**2))
        if rmsy==0.0: rmsy=1.0    

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
        nelements = npix*norder
        if covariance is True:
            cross /= nelements
            cross_error /= nelements
        else:
            cross /= rmsx*rmsy
            cross_error /= rmsx*rmsy

        
    # SIGLE-ORDER OR ONE template with MULTIPLE spectrum to cross-correlate
    #######################################################################
    else:

        # Initialize the output arrays
        cross = np.zeros((nlag,norder),dtype=float)
        cross_error = np.zeros((nlag,norder),dtype=float)
        num = np.zeros((nlag,norder),dtype=int)  # number of "good" points at this lag        
        rmsx = np.zeros(norder,dtype=float)
        rmsy = np.zeros(norder,dtype=float)    
    
        # Loop over orders
        for i in range(norder):
            # Loop over lag points
            for k in range(nlag):
                # Note the reversal of the variables for negative lags.
                if lag[k]>0:
                    cross[k,i] = np.sum(xd[0:nx-lag[k],i] * yd[lag[k]:,i])
                    num[k,i] = np.sum(fx[0:nx-lag[k],i] * fy[lag[k]:,i]) 
                    if yerr is not None:
                        cross_error[k,i] = np.sum( (xd[0:nx-lag[k],i] * yderr[lag[k]:,i])**2 )
                else:
                    cross[k,i] =  np.sum(yd[0:nx+lag[k],i] * xd[-lag[k]:,i])
                    num[k,i] = np.sum(fy[0:nx+lag[k],i] * fx[-lag[k]:,i])
                    if yerr is not None:
                        cross_error[k,i] = np.sum( (yderr[0:nx+lag[k],i] * xd[-lag[k]:,i])**2 )
                    
            if (npix>2):
                rmsx[i] = np.sum(xd[fx[:,i],i]**2)
                if (rmsx[i]==0): rmsx[i]=1.0
                rmsy[i] = np.sum(yd[fy[:,i],i]**2)
                if (rmsy[i]==0): rmsy[i]=1.0
            else:
                rmsx[i] = 1.0
                rmsy[i] = 1.0
    
        # Both X and Y are 2D, sum data from multiple orders
        if (nxorder>1) & (nyorder>1):
            cross = np.sum(cross,axis=1).reshape(nlag,1)
            cross_error= np.sum(cross_error,axis=1).reshape(nlag,1)
            num = np.sum(num,axis=1).reshape(nlag,1)
            rmsx = np.sqrt(np.sum(rmsx,axis=0)).reshape(1)
            rmsy = np.sqrt(np.sum(rmsy,axis=0)).reshape(1)
            norder = 1
            nelements = npix*norder
        else:
            rmsx = np.sqrt(rmsx)
            rmsy = np.sqrt(rmsy)        
            nelements = npix
  
        # Normalizations
        for i in range(norder):
            # Normalize by number of "good" points
            cross[:,i] *= np.max(num[:,i])
            pnum = (num[:,i]>0)
            cross[pnum,i] /= num[pnum,i]  # normalize by number of "good" points
            # Take sqrt to finish adding errors in quadrature
            cross_error[:,i] = np.sqrt(cross_error[:,i])
            # normalize
            cross_error[:,i] *= np.max(num[:,i])
            cross_error[pnum,i] /= num[pnum,i]

            # Divide by N for covariance, or divide by variance for correlation.
            if covariance is True:
                cross[:,i] /= nelements
                cross_error[:,i] /= nelements
            else:
                cross[:,i] /= rmsx[i]*rmsy[i]
                cross_error[:,i] /= rmsx[i]*rmsy[i]

        # Flatten to 1D if norder=1
        if norder==1:
            cross = cross.flatten()
            cross_error = cross_error.flatten()
            
    if yerr is not None: return cross, cross_error
    return cross


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
    .. code-block:: python

         out = apxcorr(wave,tempspec,spec,err)
    
    """


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
    if ((nlag % 2) == 0): nlag +=1  # make sure nlag is odd
    dlag = 1
    minlag = -np.int(np.ceil(nlag/2))
    lag = np.arange(nlag)*dlag+minlag+1

    # Initialize the output structure
    outstr = np.zeros(1,dtype=xcorr_dtype(nlag))
    outstr["xshift"] = np.nan
    outstr["xshifterr"] = np.nan
    outstr["vrel"] = np.nan
    outstr["vrelerr"] = np.nan
    outstr["chisq"] = np.nan

    wobs = wave.copy()
    nw = len(wobs)
    spec = obsspec.copy()
    err = obserr.copy()
    template = tempspec.copy()

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
    
    nindobs = np.sum(np.isfinite(spec) == True)  # only finite values, in case any NAN
    nindtemp = np.sum(np.isfinite(template) == True)  # only finite values, in case any NAN    
    if (nindobs>0) & (nindtemp>0):
        # Cross-Correlation
        #------------------
        # Calculate the CCF uncertainties using propagation of errors
        # Make median filtered error array
        #   high error values give crazy values in ccferr
        obserr1 = err.copy()
        if err.ndim==1:
            bderr = ((obserr1 > 1) | (obserr1 <= 0.0))
            nbderr = np.sum(bderr)
            ngderr = np.sum((bderr==False))
            if (nbderr > 0) & (ngderr > 1): obserr1[bderr]=np.median([obserr1[(bderr==False)]])
            obserr1 = median_filter(obserr1,51)
        if err.ndim==2:
            for i in range(obserr1.shape[1]):
                oerr1 = obserr1[:,i]
                bderr = ((oerr1 > 1) | (oerr1 <= 0.0))
                nbderr = np.sum(bderr)
                ngderr = np.sum((bderr==False))
                if (nbderr > 0) & (ngderr > 1): oerr1[bderr]=np.median([oerr1[(bderr==False)]])
                oerr1 = median_filter(oerr1,51)
                obserr1[:,i] = oerr1

        # Run the cross-correlation
        ccf, ccferr = ccorrelate(template,spec,lag,obserr1)
        
        # Apply flat-topped Gaussian prior with unit amplitude
        #  add a broader Gaussian underneath so the rest of the
        #   CCF isn't completely lost
        if prior is not None:
            ccf *= np.exp(-0.5*(((lag-prior[0])/prior[1])**4))*0.8+np.exp(-0.5*(((lag-prior[0])/150)**2))*0.2
        
    else:   # no good pixels
        ccf = np.zeros(len(lag),float)
        if (errccf is True): ccferr=ccf

    # Remove the median
    ccf -= np.median(ccf)
    
    # Best shift
    best_shiftind0 = np.argmax(ccf)
    best_xshift0 = lag[best_shiftind0]
    #temp = shift( tout, best_xshift0)
    temp = np.roll(template, best_xshift0, axis=0)

    # Find Chisq for each synthetic spectrum
    gdmask = (np.isfinite(spec)==True) & (np.isfinite(temp)==True) & (spec>0.0) & (err>0.0) & (err < 1e5)
    ngdpix = np.sum(gdmask)
    if (ngdpix==0):
        raise Exception('Bad spectrum')
    chisq = np.sqrt( np.sum( (spec[gdmask]-temp[gdmask])**2/err[gdmask]**2 )/ngdpix )
    
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
    pars0, cov0 = dln.gaussfit(lag,ccf_diff,estimates0,ccferr,bounds=(lbounds0,ubounds0))
    perror0 = np.sqrt(np.diag(cov0))

    # Fit the width
    #  keep height, center and constant constrained
    estimates1 = pars0
    estimates1[1] = best_xshift
    lbounds1 = [0.5*estimates1[0], best_xshift-4, 0.3*estimates1[2], dln.lt(np.min(ccf_diff),dln.lt(0,estimates1[3]-0.1)) ]
    ubounds1 =  [1.5*estimates1[0], best_xshift+4, 1.5*estimates1[2], dln.gt(np.max(ccf_diff)*0.5,estimates1[3]+0.1) ]
    lo1 = np.int(dln.gt(np.floor(best_shiftind-dln.gt(estimates1[2]*2,5)),0))
    hi1 = np.int(dln.lt(np.ceil(best_shiftind+dln.gt(estimates1[2]*2,5)),len(lag)))
    pars1, cov1 = dln.gaussfit(lag[lo1:hi1],ccf_diff[lo1:hi1],estimates1,ccferr[lo1:hi1],bounds=(lbounds1,ubounds1))
    yfit1 = dln.gaussian(lag[lo1:hi1],*pars1)
    perror1 = np.sqrt(np.diag(cov1))
    
    # Fefit and let constant vary more, keep width constrained
    estimates2 = pars1
    estimates2[1] = dln.limit(estimates2[1],np.min(lag),np.max(lag))    # must be in range
    estimates2[3] = np.median(ccf_diff[lo1:hi1]-yfit1) + pars1[3]
    lbounds2 = [0.5*estimates2[0], dln.limit(best_xshift-dln.gt(estimates2[2],1), np.min(lag), estimates2[1]-1),
                0.3*estimates2[2], dln.lt(np.min(ccf_diff),dln.lt(0,estimates2[3]-0.1)) ]
    ubounds2 = [1.5*estimates2[0], dln.limit(best_xshift+dln.gt(estimates2[2],1), estimates2[1]+1, np.max(lag)),
                1.5*estimates2[2], dln.gt(np.max(ccf_diff)*0.5,estimates2[3]+0.1) ]
    lo2 = np.int(dln.gt(np.floor( best_shiftind-dln.gt(estimates2[2]*2,5)),0))
    hi2 = np.int(dln.lt(np.ceil( best_shiftind+dln.gt(estimates2[2]*2,5)),len(lag)))
    pars2, cov2 = dln.gaussfit(lag[lo2:hi2],ccf_diff[lo2:hi2],estimates2,ccferr[lo2:hi2],bounds=(lbounds2,ubounds2))
    yfit2 = dln.gaussian(lag[lo2:hi2],*pars2)    
    perror2 = np.sqrt(np.diag(cov2))
    
    # Refit with even narrower range
    estimates3 = pars2
    estimates3[1] = dln.limit(estimates3[1],np.min(lag),np.max(lag))    # must be in range
    estimates3[3] = np.median(ccf_diff[lo1:hi1]-yfit1) + pars1[3]
    lbounds3 = [0.5*estimates3[0], dln.limit(best_xshift-dln.gt(estimates3[2],1), np.min(lag), estimates3[1]-1),
                0.3*estimates3[2], dln.lt(np.min(ccf_diff),dln.lt(0,estimates3[3]-0.1)) ]
    ubounds3 = [1.5*estimates3[0], dln.limit(best_xshift+dln.gt(estimates3[2],1), estimates3[1]+1, np.max(lag)),
                1.5*estimates3[2], dln.gt(np.max(ccf_diff)*0.5,estimates3[3]+0.1) ]    
    lo3 = np.int(dln.gt(np.floor(best_shiftind-dln.gt(estimates3[2]*2,5)),0))
    hi3 = np.int(dln.lt(np.ceil(best_shiftind+dln.gt(estimates3[2]*2,5)),len(lag)))
    pars3, cov3 = dln.gaussfit(lag[lo3:hi3],ccf_diff[lo3:hi3],estimates3,ccferr[lo3:hi3],bounds=(lbounds3,ubounds3))
    yfit3 = dln.gaussian(lag[lo3:hi3],*pars3)    
    perror3 = np.sqrt(np.diag(cov3))

    # This seems to fix high shift/sigma errors
    if (perror3[0]>10) | (perror3[1]>10):
        dlbounds3 = [0.5*estimates3[0], -10+pars3[1], 0.01, dln.lt(np.min(ccf_diff),dln.lt(0,estimates3[3]-0.1)) ]
        dubounds3 = [1.5*estimates3[0], 10+pars3[1], 2*pars3[2], dln.gt(np.max(ccf_diff)*0.5,estimates3[3]+0.1) ]
        dpars3, dcov3 = dln.gaussfit(lag[lo3:hi3],ccf_diff[lo3:hi3],pars3,ccferr[lo3:hi3],bounds=(dlbounds3,dubounds3))
        dyfit3 = dln.gaussian(lag[lo3:hi3],*pars3)    
        perror3 = np.sqrt(np.diag(dcov3))
        
    # Final parameters
    pars = pars3
    perror = perror3
    xshift = pars[1]
    xshifterr = perror[1]
    ccpfwhm_pix = pars[2]*2.35482  # ccp fwhm in pixels
    # v = (10^(delta log(wave))-1)*c
    dwlog = np.median(dln.slope(np.log10(wave)))
    ccpfwhm = ( 10**(ccpfwhm_pix*dwlog)-1 )*cspeed  # in km/s

    # Convert pixel shift to velocity
    #---------------------------------
    # delta log(wave) = log(v/c+1)
    # v = (10^(delta log(wave))-1)*c
    dwlog = np.median(dln.slope(np.log10(wave)))
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


def normspec(spec=None,ncorder=6,fixbadpix=True,noerrcorr=False,
             binsize=0.05,perclevel=95.0,growsky=False,nsky=5):
    """
    This program normalizes a spectrum.

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

    Examples
    --------
    .. code-block:: python

         nspec,cont,masked = normspec(spec)

    """

    # Not enough inputs
    if spec is None:
        raise ValueError("""spec2 = normspec(spec,fixbadpix=fixbadpix,ncorder=ncorder,noerrcorr=noerrcorr,
                                             binsize=binsize,perclevel=perclevel)""")
    musthave = ['flux','err','mask','wave']
    for a in musthave:
        if hasattr(spec,a) is False:
            raise ValueError("spec object must have "+a)

    # Can only do 1D or 2D arrays
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
    coef1 = dln.poly_fit(xbin[gdbin],ybin[gdbin],ncorder,robust=True)
    cont1 = dln.poly(x,coef1)

    # Subtract smoothed error from it to remove the effects
    #  of noise on the continuum measurement
    if (yerr is not None) & (noerrcorr is False):
        smyerr = dln.medfilt(yerr,151)                            # first median filter
        smyerr = dln.gsmooth(smyerr,100)                          # Gaussian smoothing
        coef_err = dln.poly_fit(x,smyerr,ncorder,robust=True)     # fit with robust poly
        #poly_err = dln.poly(x,coef_err)
        #cont1 -= 2*dln.poly_err   # is this right????
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
    coef2 = dln.poly_fit(xbin2[gdbin2],ybin2[gdbin2],ncorder,robust=True)
    cont2 = dln.poly(x,coef2)

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


def spec_resid(pars,wave,flux,err,models,spec):
    """
    This helper function calculates the residuals between an observed spectrum and a Cannon model spectrum.
    
    Parameters
    ----------
    pars : array
      Input parameters [teff, logg, feh, rv].
    wave : array
      Wavelength array for observed spectrum.
    flux : array
       Observed flux array.
    err : array
        Uncertainties in the observed flux.
    models : list of Cannon models
        List of Cannon models to use
    spec : Spec1D
        The observed spectrum.  Needed to run cannon.model_spectrum().

    Returns
    -------
    resid : array
         Array of residuals between the observed flux array and the Cannon model spectrum.

    Example
    -------
    .. code-block:: python

         resid = spec_resid(pars,wave,flux,err,models,spec)

    """
    #m = cannon.model_spectrum(models,spec,teff=pars[0],logg=pars[1],feh=pars[2],rv=pars[3])
    m = models(teff=pars[0],logg=pars[1],feh=pars[2],rv=pars[3])
    if m is None:
        return np.repeat(1e30,len(flux))
    resid = (flux-m.flux.flatten())/err
    return resid


def printpars(pars,parerr=None,names=None,units=None):
    """
    Print out the parameters/labels with optionally uncertainties and units.
    One parameter per line of output.

    Parameters
    ----------
    pars : list or numpy array
       List or array of parameter values to print out.  The default is
         TEFF, LOGG, FE_H and VREL.
    parerr : list or numpy array, optional
       List or array of parameter uncertainties to print out. No uncertainties
         are printed by default.
    names : list, optional
       List of parameter names.  The default is ['Teff','logg','[Fe/H]','Vrel'] if
        4 parameters are input or ['Teff','logg','[Fe/H]','[alpha/Fe]','Vrel'] if
        5 parameteres are input.
    units : list, optional
       List of parameter units.

    Returns
    -------
    Parameter values are printed to the screen, one parameter per line.

    Example
    -------
    .. code-block:: python

         printpars(pars,parerr,names,names)

    """

    print = utils.getprintfunc() # Get print function to be used locally, allows for easy logging

    npars = len(pars)
    if names is None:
        if npars==3:
            names = ['Teff','logg','[Fe/H]']
            units = ['K','','']
        if npars==4:
            names = ['Teff','logg','[Fe/H]','Vrel']
            units = ['K','','','km/s']
        if npars==5:
            names = ['Teff','logg','[Fe/H]','[alpha/Fe]','Vrel']
            units = ['K','','','','km/s']
    if units is None:
        units = [' ' for l in pars]
            
    for i in range(npars):
        if parerr is None:
            err = None
        else:
            err = parerr[i]
        if err is not None:
            print('%-6s =  %8.2f +/- %6.3f %-5s' % (names[i],pars[i],err,units[i]))
        else:
            print('%-6s =  %8.2f %-5s' % (names[i],pars[i],units[i]))


######################### PAYNE PROGRAMS ###########################


def emcee_lnlike_payne(theta, x, y, yerr, sp):
    """
    This helper function calculates the log likelihood for fit_mcmc_payne().
    
    Parameters
    ----------
    theta : array
      Input parameters [teff, logg, feh, rv].
    x : array
      Array of x-values for y.  Not really used.
    y : array
       Observed flux array.
    yerr : array
        Uncertainties in the observed flux.
    sp : PayneSpecFitter
       Special class for fitting Payne models to data.

    Returns
    -------
    lnlike : float
         The log likelihood value.

    """
    mflux = sp.model(x,*theta)
    inv_sigma2 = 1.0/yerr**2
    return -0.5*(np.sum((y-mflux)**2*inv_sigma2))


def emcee_lnprior_payne(theta, sp):
    """
    This helper function calculates the log prior for fit_mcmc_payne().
    It's a flat/uniform prior across the stellar parameter space covered by the
    Payne model.
    
    Parameters
    ----------
    theta : array
      Input parameters [teff, logg, feh, rv, etc.].
    sp : PayneSpecFitter
       Special class for fitting Payne models to data.

    Returns
    -------
    lnprior : float
         The log prior value.

    """
    lbounds,ubounds = sp.bounds
    nfitparams = len(sp.fitparams)
    
    inside = True
    for i in range(nfitparams):
        inside &= (theta[i]>=lbounds[i]) & (theta[i]<=ubounds[i])
    if inside:
        return 0.0
    return -np.inf

def emcee_lnprob_payne(theta, x, y, yerr, sp):
    """
    This helper function calculates the log probability for fit_mcmc_payne().
    
    Parameters
    ----------
    theta : array
      Input parameters [teff, logg, feh, rv, etc.].
    x : array
      Array of x-values for y.  Not really used.
    y : array
       Observed flux array.
    yerr : array
        Uncertainties in the observed flux.
    sp : PayneSpecFitter
       Special class for fitting Payne models to data.

    Returns
    -------
    lnprob : float
         The log probability value, which is the sum of the log prior and the
         log likelihood.

    """
    lp = emcee_lnprior_payne(theta,sp)
    if not np.isfinite(lp):
        return -np.inf
    return lp + emcee_lnlike_payne(theta, x, y, yerr, sp)


def fit_xcorrgrid_payne(spec,model=None,samples=None,verbose=False,maxvel=1000.0):
    """
    Fit spectrum using cross-correlation with models sampled in the parameter space.
    
    Parameters
    ----------
    spec : Spec1D object
         The observed spectrum to match.
    model : Payne model, optional
         Payne model to use.  The default is to load all of the Payne
         models in the data/ directory and use those.
    samples : numpy structured array, optional
         Catalog of teff/logg/feh/alphafe parameters to use when sampling the parameter space.
    verbose : bool, optional
         Verbose output of the various steps.  This is False by default.
    maxvel : float, optional
         The maximum velocity to probe in the cross-correlation.  The default is 1000 km/s.

    Returns
    -------
    out : numpy structured array
         The output structured array of the final derived RVs, stellar parameters and errors.
    bmodel : Spec1D object
         The best-fitting Payne model spectrum (as Spec1D object).

    Example
    -------

    .. code-block:: python

         out, bmodel = fit_xcorrgrid_payne(spec)

    """

    print = utils.getprintfunc() # Get print function to be used locally, allows for easy logging
    
    # Check that the samples input has the right columns
    if samples is not None:
        for n in ['teff','logg','fe_h','alpha_h']:
            try:
                dum = samples[n]
            except:
                raise ValueError(n+' not found in input SAMPLES')
    
    # Step 1: Prepare the spectrum
    #-----------------------------
    # normalize and mask spectrum
    if spec.normalized is False:
        spec = utils.specprep(spec)

    # Step 2: Load and prepare the Cannon models
    #-------------------------------------------
    if model is None:
        model = payne.load_models()
    # Prepare if necessary
    if model.prepared is False: model.prepare(spec)
        
    # Step 3: put on logarithmic wavelength grid
    #-------------------------------------------
    wavelog = utils.make_logwave_scale(spec.wave,vel=0.0)  # get new wavelength solution
    obs = spec.interp(wavelog)
    # The LSF information will not be correct if using Gauss-Hermite, it uses a Gaussian approximation
    # it's okay because the "model" is prepared for the original spectra (above)
    
    # Step 4: get initial RV using cross-correlation with rough sampling of Teff/logg parameter space
    #------------------------------------------------------------------------------------------------
    dwlog = np.median(dln.slope(np.log10(wavelog)))
    # vrel = ( 10**(xshift*dwlog)-1 )*cspeed
    maxlag = np.int(np.ceil(np.log10(1+maxvel/cspeed)/dwlog))
    maxlag = np.maximum(maxlag,50)
    if samples is None:
        teff = [3500.0, 4000.0, 5000.0, 6000.0, 7500.0, 15000.0, 25000.0, 40000.0,  3500.0, 4300.0, 4700.0, 5200.0]
        logg = [4.8, 4.8, 4.6, 4.4, 4.0, 4.0, 4.0, 4.0,  0.5, 1.0, 2.0, 3.0]        
        feh = [-1.5, -0.5, -1.5, -0.5]
        alphafe = [0.0, 0.0, 0.3, 0.3]
        dt = np.dtype([('teff',float),('logg',float),('feh',float),('alphafe',float)])
        samples = np.zeros(len(teff)*len(feh),dtype=dt)
        nteff = len(teff)
        for i in range(len(feh)):
            samples['teff'][i*nteff:(i+1)*nteff] = teff
            samples['logg'][i*nteff:(i+1)*nteff] = logg
            samples['feh'][i*nteff:(i+1)*nteff] = feh[i]
            samples['alphafe'][i*nteff:(i+1)*nteff] = alphafe[i] 
    outdtype = np.dtype([('xshift',np.float32),('vrel',np.float32),('vrelerr',np.float32),('ccpeak',np.float32),('ccpfwhm',np.float32),
                         ('chisq',np.float32),('teff',np.float32),('logg',np.float32),('feh',np.float32),('alphafe',np.float32)])
    outstr = np.zeros(len(samples),dtype=outdtype)
    if verbose is True: print('  TEFF    LOGG   FEH   ALPHAFE    VREL  CCPEAK  CHISQ')
    for i in range(len(samples)):
        inpdict = {'teff':samples['teff'][i],'logg':samples['logg'][i],'fe_h':samples['feh'][i],
                   'alpha_h':samples['alphafe'][i]+samples['feh'][i],'rv':0.0}
        labels = model.mklabels(inpdict)
        m = model(labels,wave=wavelog)
        outstr1 = specxcorr(m.wave,m.flux,obs.flux,obs.err,maxlag)
        #if outstr1['chisq'] > 1000:
        #    import pdb; pdb.set_trace()
        if verbose is True:
            print('%7.0f  %5.2f  %5.2f  %5.2f  %8.2f  %5.2f  %5.2f' % (samples['teff'][i],samples['logg'][i],samples['feh'][i],
                                                                       samples['alphafe'][i],outstr1['vrel'][0],outstr1['ccpeak'][0],
                                                                       outstr1['chisq'][0]))
        for n in ['xshift','vrel','vrelerr','ccpeak','ccpfwhm','chisq']: outstr[n][i] = outstr1[n]
        outstr['teff'][i] = samples['teff'][i]
        outstr['logg'][i] = samples['logg'][i]
        outstr['feh'][i] = samples['feh'][i]
        outstr['alphafe'][i] = samples['alphafe'][i]
        #import pdb; pdb.set_trace()
    # Get best fit
    bestind = np.argmin(outstr['chisq'])    
    beststr = outstr[bestind]
    bestlabels = model.mklabels({'teff':beststr['teff'],'logg':beststr['logg'],'fe_h':beststr['feh'],
                                 'alpha_h':beststr['alphafe']+beststr['feh'],'rv':beststr['vrel']})
    bestmodel = model(bestlabels)
    
    if verbose is True:
        print('Initial RV fit:')
        printpars([beststr['teff'],beststr['logg'],beststr['feh'],beststr['alphafe'],beststr['vrel']],
                  [None,None,None,None,beststr['vrelerr']])
        print('chisq = %5.2f' % beststr['chisq'])
        
    return beststr, bestmodel

def make_payne_initlabels(labels):
    """ Make initial guesses for Payne labels."""

    labels = np.char.array(labels).upper()
    
    # Initializing the labels array
    nlabels = len(labels)
    initpars = np.zeros(nlabels,float)
    initpars[labels=='TEFF'] = 5000.0
    initpars[labels=='LOGG'] = 3.5
    initpars[labels.endswith('_H')] = 0.0
    # Vmicro/Vturb=2.0 km/s by default
    initpars[(labels=='VTURB') | (labels=='VMICRO')] = 2.0
    # All abundances, VSINI, VMACRO, RV = 0.0
            
    return initpars


def make_payne_bounds(labels,initpars=None):
    """ Make upper and lower bounds for Payne labels."""

    if initpars is None:
        initpars = make_payne_initlabels(labels)
    nlabels = len(labels)
    lbounds = np.zeros(nlabels,np.float64)
    ubounds = np.zeros(nlabels,np.float64)

    # Initial guesses and bounds for the fitted parameters
    for i,par in enumerate(labels):
        if par.upper()=='TEFF':
            lbounds[i] = np.maximum(initpars[i]-2000,3000)
            ubounds[i] = initpars[i]+2000
        if par.upper()=='LOGG':
            lbounds[i] = np.maximum(initpars[i]-2,0)
            ubounds[i] = np.minimum(initpars[i]+2,5)
        if par.upper()=='VTURB':
            lbounds[i] = np.maximum(initpars[i]-2,0)
            ubounds[i] = initpars[i]+2
        if par.upper().endswith('_H'):
            lbounds[i] = np.maximum(initpars[i]-0.75,-2.5)
            ubounds[i] = np.minimum(initpars[i]+0.75,0.5)
        if par.upper()=='FE_H':
            lbounds[i] = -2.5
            ubounds[i] = 0.5
        if par.upper()=='VSINI':
            lbounds[i] = np.maximum(initpars[i]-20,0)
            ubounds[i] = initpars[i]+50
        if par.upper()=='VMACRO':
            lbounds[i] = np.maximum(initpars[i]-2,0)
            ubounds[i] = initpars[i]+2
        if par.upper()=='RV':
            lbounds[i] = -1000.0
            ubounds[i] = 1000.0
            
    bounds = (lbounds,ubounds)
    
    return bounds


def fit_lsq_payne(spec,model=None,initpar=None,fitparams=None,fixparams={},verbose=False):
    """
    Least Squares fitting with forward modeling of the spectrum.
    
    Parameters
    ----------
    spec : Spec1D object
         The observed spectrum to match.
    model : Payne model, optional
         Payne model to use.  The default is to load all of the Payne
         models in the data/ directory and use those.
    initpar : dictionary or array, optional
         Initial estimates for parameters.  If it is an array, then it is
         assumbed that these are for the FITPARAMS parameters.  If it is a
         dictionary, then the parameters are clear and they do not need to
         cover all FITPARAMS parameters.  Optional.
    fitparams : list of labels, optional
         List of Payne parameter/label names to fit. Optional.
         The default values are ['TEFF','LOGG','FE_H','ALPHA_H','RV'].
    fixparams : dict, optional
         Dictionary of parameters to hold fixed.
    verbose : bool, optional
         Verbose output of the various steps.  This is False by default.

    Returns
    -------
    out : numpy structured array
         The output structured array of the final derived RVs, stellar parameters and errors.
    bmodel : Spec1D object
         The best-fitting Payne model spectrum (as Spec1D object).

    Example
    -------

    .. code-block:: python

         out, bmodel = fit_lsq_payne(spec,model)

    """

    print = utils.getprintfunc() # Get print function to be used locally, allows for easy logging    
    
    # Prepare the spectrum
    #-----------------------------
    # normalize and mask spectrum
    if spec.normalized is False:
        spec = utils.specprep(spec)

    # Load and prepare the Cannon models
    #-------------------------------------------
    if model is None:
        model = payne.load_models()
    # Prepare if necessary
    if model.prepared is False: model.prepare(spec)

    # Fitting parameters
    if fitparams is None:
        fitparams = ['TEFF','LOGG','FE_H','ALPHA_H','RV']
    else:
        fitparams = [f.upper() for f in fitparams]  # all CAPS
    nfitparams = len(fitparams)
    nfixparams = len(fixparams)
    nallparams = nfitparams+nfixparams
    
    # Get initial estimates
    #  this does not need to include initial guess for all fitparams
    #  that's taken care of below
    if initpar is not None and isinstance(initpar,dict) is False:
        # Array input
        #  must be the initial parameters for FITPARAMS        
        if len(initpar)!=len(fitparams):
            raise Exception("INITPAR must have same number of elements as FITPARAMS")
        initpar = dict(zip(fitparams,initpar))  # make dictionary
            
    # Make initial parameters for all labels
    allinitpar = make_payne_initlabels(fitparams)
    if initpar is not None:
        for name in initpar.keys():
            ind, = np.where(np.char.array(fitparams)==name)
            if len(ind)>0:
                allinitpar[ind[0]] = initpar[name]
            
    # Initialize spectral fitter
    sp = payne.PayneSpecFitter(spec,model,fitparams,fixparams,verbose=False)

    # Calculate the bounds
    bounds = sp.mkbounds(fitparams,allinitpar)
    
    # Use curve_fit
    tol = 5e-5  # 5e-4
    dx_lim = sp.mkdxlim(fitparams)
    lspars, lscov = curve_fit(sp.model,spec.wave.flatten(),spec.flux.flatten(), #dx_lim=dx_lim,
                              sigma=spec.err.flatten(),p0=allinitpar,bounds=bounds,jac=sp.jac)
    #print(sp.nfev,sp.njac)
    
    # If it hits a boundary then the solution won't change much compared to initpar
    # setting absolute_sigma=True gives crazy low lsperror values
    lsperror = np.sqrt(np.diag(lscov))
    
    if verbose is True:
        print('Least Squares RV and stellar parameters:')
        printpars(lspars,names=fitparams)

    
    if nfixparams>0:
        all_labels = fitparams+list(fixparams.keys())
        all_pars = lspars + fixparams.values()
    else:
        all_labels = fitparams
        all_pars = lspars
    bestlabels = model.mklabels(dict(zip(all_labels,all_pars)))    
    lsmodel = model(bestlabels)
    
    lschisq = np.sqrt(np.sum(((spec.flux-lsmodel.flux)/spec.err)**2)/(spec.npix*spec.norder))
    if verbose is True: print('chisq = %5.2f' % lschisq)

    # Put it into the output structure
    dtype = np.dtype([('labels','<U100',nfitparams),('pars',float,nfitparams),('parerr',float,nfitparams),
                      ('parcov',float,(nfitparams,nfitparams)),('chisq',float)])
    out = np.zeros(1,dtype=dtype)
    out['labels'] = fitparams
    out['pars'] = lspars
    out['parerr'] = lsperror
    out['parcov'] = lscov
    out['chisq'] = lschisq
    
    return out, lsmodel

def mcmc_delta_payne(labels):
    """ Get MCMC deltas for the walkers."""
    delta = np.zeros(len(labels),float)
    for i in range(len(labels)):
        name = labels[i]
        if name=='TEFF':
            step = 50.0
        elif name=='RV':
            step = 5
        elif name=='VROT':
            step = 5
        elif name=='VMICRO':
            step = 1.0
        elif name.endswith('_H'):
            step = 0.1
        else:
            step = 0.1
        delta[i] = step
    return delta


def fit_mcmc_payne(spec,model=None,fitparams=None,fixparams={},initpar=None,steps=100,
                   cornername=None,verbose=False):
    """
    Fit the spectrum with MCMC.
    
    Parameters
    ----------
    spec : Spec1D object
         The observed spectrum to match.
    model: Payne model, optional
         Payne model to use.  The default is to load the Payne model
         in the data/ directory and use those.
    fitparams : list of labels, optional
         List of Payne parameter/label names to fit. Optional.
         The default values are ['TEFF','LOGG','FE_H','ALPHA_H','RV'].
    fixparams : dict, optional
         Dictionary of parameters to hold fixed.
    initpar : numpy array, optional
         Initial estimate for [teff, logg, feh, RV], optional.
    steps : int, optional
         Number of steps to use.  Default is 100.
    cornername : string, optional
         Output filename for the corner plot.  If a corner plot is requested, then the
         minimum number of steps used is 500.
    verbose : bool, optional
         Verbose output of the various steps.  This is False by default.

    Returns
    -------
    out : numpy structured array
         The output structured array of the final derived RVs, stellar parameters and errors.
    bmodel : Spec1D object
         The best-fitting Cannon model spectrum (as Spec1D object).

    Example
    -------

    .. code-block:: python

         out, bmodel = fit_mcmc_payne(spec)

    """

    print = utils.getprintfunc() # Get print function to be used locally, allows for easy logging
    
    # Prepare the spectrum
    #-----------------------------
    # normalize and mask spectrum
    if spec.normalized is False:
        spec = utils.specprep(spec)

    # Fitting parameters
    if fitparams is None:
        fitparams = ['TEFF','LOGG','FE_H','ALPHA_H','RV'] 
    nfitparams = len(fitparams)
    nfixparams = len(fixparams)

    # Load the Payne model
    #-----------------------
    if model is None: model = payne.load_models()
    model.prepare(spec)
    
    # Initial estimates
    if initpar is None:
        initpar = make_payne_initlabels(fitparams)
            
    # Initialize spectral fitter
    sp = payne.PayneSpecFitter(spec,model,fitparams,fixparams,verbose=False)

    # Calculate the bounds
    bounds = sp.mkbounds(fitparams,initpar)
    sp.bounds = bounds

    # Set up the MCMC sampler
    ndim, nwalkers = nfitparams, 20
    delta = mcmc_delta_payne(fitparams)
    pos = [initpar + delta*np.random.randn(ndim) for i in range(nwalkers)]        
    sampler = emcee.EnsembleSampler(nwalkers, ndim, emcee_lnprob_payne,
                                    args=(spec.wave.flatten(), spec.flux.flatten(), spec.err.flatten(), sp))
    if cornername is not None: steps=np.maximum(steps,500)  # at least 500 steps
    
    # Run MCMC sampler
    if verbose: print('Running MCMC')
    out = sampler.run_mcmc(pos, steps)
    samples = sampler.chain[:, np.int(steps/2):, :].reshape((-1, ndim))

    # Get the median and stddev values
    pars = np.zeros(ndim,float)
    parerr = np.zeros(ndim,float)
    if verbose is True: print('MCMC values:')
    for i in range(ndim):
        t=np.percentile(samples[:,i],[16,50,84])
        pars[i] = t[1]
        parerr[i] = (t[2]-t[0])*0.5
    if verbose is True: printpars(pars,parerr,names=fitparams)
        
    # The maximum likelihood parameters
    bestind = np.unravel_index(np.argmax(sampler.lnprobability),sampler.lnprobability.shape)
    pars_ml = sampler.chain[bestind[0],bestind[1],:]

    # Get best model
    if nfixparams>0:
        all_labels = fitparams+list(fixparams.keys())
        all_pars = pars + fixparams.values()
    else:
        all_labels = fitparams
        all_pars = pars
    bestlabels = model.mklabels(dict(zip(all_labels,all_pars)))    
    mcmodel = model(bestlabels)
    mcchisq = np.sqrt(np.sum(((spec.flux-mcmodel.flux)/spec.err)**2)/(spec.npix*spec.norder))

    # Put it into the output structure
    dtype = np.dtype([('pars',float,ndim),('pars_ml',float,ndim),('parerr',float,ndim),('chisq',float)])
    out = np.zeros(1,dtype=dtype)
    out['pars'] = pars
    out['pars_ml'] = pars_ml    
    out['parerr'] = parerr
    out['chisq'] = mcchisq
    
    # Corner plot
    if cornername is not None:
        backend = matplotlib.rcParams['backend']        
        matplotlib.use('Agg')
        fig = corner.corner(samples, labels=fitparams, truths=pars)
        plt.savefig(cornername)
        plt.close(fig)
        print('Corner plot saved to '+cornername)
        matplotlib.use(backend)  # back to the original backend
        
    return out,mcmodel
 

def fit_payne(spectrum,model=None,fitparams=None,fixparams={},estimates=None,
              verbose=False,figfile=None,mcmc=False,cornername=None,nthreads=None,
              notweak=False,tpoly=False,tpolyorder=3):
    """
    Fit the spectrum.  Find the best RV and stellar parameters using the Payne model.

    Parameters
    ----------
    spectrum : Spec1D object
         The observed spectrum to match.
    model: Payne model, optional
         Payne model to use.  The default is to load the Payne model
         in the data/ directory and use those.
    fitparams : list of labels, optional
         List of Payne parameter/label names to fit. Optional.
         The default values are ['TEFF','LOGG','FE_H','ALPHA_H','RV'].
    fixparams : dict, optional
         Dictionary of parameters to hold fixed.
    estimates : dict, optional
         Dictionary of initial guesses.
    verbose : bool, optional
         Verbose output of the various steps.  This is False by default.
    mcmc : bool, optional
         Run Markov Chain Monte Carlo (MCMC) to get improved parameter uncertainties.
         This is False by default.
    figfile : string, optional
         The filename for a diagnostic plot showing the observed spectrum, model
         spectrum and the best-fit parameters.
    cornername : string, optional
         The filename for a "corner" plot showing the posterior distributions from
         the MCMC run.
    nthreads : int, optional
         The number of threads to use.  By default the number of threads is not limited.
    notweak : boolean, optional
         Don't tweak the observed continuum using the model.  Default is False.
    tpoly : boolean, optional
         Use a low-order polynomial fit for continuum tweaking.  Default is False,
           it uses Gaussian smoothing instead.
    tpolyorder : int, optional
         Polynomial order to use for continuum tweaking if tpoly=True.
           Default is 3 (cubic).

    Returns
    -------
    out : numpy structured array
         The output structured array of the final derived RVs, stellar parameters and errors.
    model : Spec1D object
         The best-fitting Payne model spectrum (as Spec1D object).
    specm : Spec1D object
         The observed spectrum with discrepant and outlier pixels masked.

    Example
    -------

    .. code-block:: python

         out, model = fit_payne(spec)

    """

    print = utils.getprintfunc() # Get print function to be used locally, allows for easy logging
    
    # Set threads
    if nthreads is not None:
        os.environ["OMP_NUM_THREADS"] = str(nthreads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
        os.environ["MKL_NUM_THREADS"] = str(nthreads)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(nthreads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(nthreads)    
    
    t0 = time.time()

    # Make internal copy
    spec = spectrum.copy()

    # Fitting parameters
    if fitparams is None:
        fitparams = ['TEFF','LOGG','FE_H','ALPHA_H','RV']
    else:
        fitparams = [f.upper() for f in fitparams]
        
    if verbose: print('Fitting Payne model to spectrum with parameters: '+', '.join(fitparams))        
    nfitparams = len(fitparams)
    nfixparams = len(fixparams)
    
    # Step 1: Prepare the spectrum
    #-----------------------------
    # Normalize and mask the spectrum
    spec = utils.specprep(spec)         
    # Mask out any large positive outliers, e.g. badly subtracted sky lines
    specm = utils.maskoutliers(spec,verbose=verbose)
    
    # Step 2: Load the Payne model
    #------------------------------
    if model is None: model = payne.load_models()
    model.prepare(specm)
    
    # Check the input labels against the Paybe model labels
    fitparams = payne.check_params(model,fitparams)
    if nfixparams>0:
        fixparams = payne.check_params(model,fixparams)    
        
    # Step 3: Get initial RV using cross-correlation with rough sampling of Teff/logg parameter space
    #------------------------------------------------------------------------------------------------
    doxcorr = True
    if estimates is not None:
        uest = {k.upper():estimates[k] for k in estimates.keys()}
        ukeys = uest.keys()
        if 'TEFF' in ukeys and 'LOGG' in ukeys and 'FE_H' in ukeys and 'RV' in ukeys:
            doxcorr = False
    if doxcorr:
        beststr, xmodel = fit_xcorrgrid_payne(specm,model,verbose=verbose,maxvel=1000.0)  
    else:
        beststr = {'teff':uest['TEFF'],'logg':uest['LOGG'],'feh':uest['FE_H'],'alphafe':0.0,'vrel':uest['RV']}
        if 'ALPHA_FE' in ukeys:
            beststr['alphafe'] = uest['ALPHA_FE']
            
    # Step 4: Least Squares fitting with forward modeling
    #----------------------------------------------------
    # Tweak the continuum
    #specm = tweakcontinuum(specm,xmodel)
    # Initial estimates
    initpar = {'TEFF':beststr['teff'],'LOGG':beststr['logg'],'FE_H':beststr['feh'],
               'ALPHA_H':beststr['alphafe']+beststr['feh'],'RV':beststr['vrel']}
    if estimates is not None:
        if verbose: print('Using input estimates: '+str(estimates))
        for k in ['TEFF','LOGG','FE_H','ALPHA_FE','RV']:
            if k in ukeys:
                initpar[k] = uest[k]

                # Run nonlinear least squares
    lsout0, lsmodel0 = fit_lsq_payne(specm,model,initpar=initpar,fitparams=fitparams,
                                   fixparams=fixparams,verbose=verbose)
    lspars0 = lsout0['pars'][0]
    lsperror0 = lsout0['parerr'][0]    

    # Tweak the continuum normalization
    if notweak is False:
        specm = tweakcontinuum(specm,lsmodel0,usepoly=tpoly,polyorder=tpolyorder)
        # Mask out very discrepant pixels when compared to the best-fit model
        specm = utils.maskdiscrepant(specm,lsmodel0,verbose=verbose)  
    
        # Refit with Payne
        lsout, lsmodel = fit_lsq_payne(specm,model,initpar=dict(zip(fitparams,lspars0)),
                                       fitparams=fitparams,fixparams=fixparams,verbose=verbose)
        lspars = lsout['pars'][0]
        lsperror = lsout['parerr'][0] 
    else:
        lsout = lsout0
        lsmodel = lsmodel0
        lspars = lspars0
        lsperror = lsperror0
        
        
    # Step 5: Run fine grid in RV, forward modeling
    #----------------------------------------------
    maxv = 10.0
    rvind, = np.where(np.char.array(fitparams)=='RV')[0]
    vel = dln.scale_vector(np.arange(31),lspars[rvind]-maxv,lspars[rvind]+maxv)
    chisq = np.zeros(len(vel))
    tinput = dict(zip(fitparams,lsout['pars'][0].copy()))
    if nfixparams>0:  # add fixed parameters
        tinput.update(fixparams)
    for i,v in enumerate(vel):
        tinput['RV'] = v
        tlabels = model.mklabels(tinput)
        m = model(tlabels)
        chisq[i] = np.sqrt(np.sum(((specm.flux-m.flux)/specm.err)**2)/(specm.npix*specm.norder))
    vel2 = dln.scale_vector(np.arange(300),lspars[rvind]-maxv,lspars[rvind]+maxv)
    chisq2 = dln.interp(vel,chisq,vel2)
    bestind = np.argmin(chisq2)
    finerv = vel2[bestind]
    finechisq = chisq2[bestind]
    if verbose is True:
        print('Fine grid best RV = %5.2f km/s' % finerv)
        print('chisq = %5.2f' % finechisq)
    
    # Final parameters and uncertainties (so far)
    fpars = lsout['pars'][0]
    fperror = lsout['parerr'][0]
    fpars[rvind] = finerv
    fchisq = finechisq
    finput = dict(zip(fitparams,fpars.copy()))
    finput['RV'] = finerv
    flabels = model.mklabels(finput)
    fmodel = model(flabels)
    
    # Step 6: MCMC
    #--------------
    if (mcmc is True) | (cornername is not None):
        mcout, mcmodel = fit_mcmc_payne(specm,model,fitparams=fitparams,fixparams=fixparams,
                                        initpar=fpars,verbose=verbose,cornername=cornername)
        # Use these parameters
        fpars = mcout['pars'][0]
        fperror = mcout['parerr'][0]
        fchisq = mcout['chisq'][0]
        fmodel = mcmodel
        
    # Construct the output
    #---------------------
    vrel = fpars[rvind]
    vrelerr = fperror[rvind]
    if specm.bc is None:
        bc = specm.barycorr()
    else:
        bc = specm.bc
    vhelio = vrel + bc
    if verbose is True:
        print('Final parameters:')
        printpars(fpars,fperror,names=fitparams)
        print('Vhelio = %6.2f +/- %5.2f km/s' % (vhelio,vrelerr))
        print('BC = %5.2f km/s' % bc)
        print('chisq = %5.2f' % fchisq)
        
    dtlist = [('vhelio',np.float32),('vrel',np.float32),('vrelerr',np.float32)]
    for k in range(nfitparams):
        name = fitparams[k].lower()
        if name != 'rv':  # Skip RV
            dtlist += [(name,np.float32),(name+'err',np.float32)]
    dtlist += [('chisq',np.float32),('bc',np.float32)]
    dtype = np.dtype(dtlist)
    out = np.zeros(1,dtype=dtype)
    out['vhelio'] = vhelio
    out['vrel'] = vrel
    out['vrelerr'] = vrelerr    
    for k in range(nfitparams):
        name = fitparams[k].lower()
        if name != 'rv':  # Skip RV
            out[name] = fpars[k]
            out[name+'err'] = fperror[k]
    out['chisq'] = fchisq
    out['bc'] = bc
    
    # Make diagnostic figure
    if figfile is not None:
        # Apply continuum tweak to original spectrum as well
        cratio = specm.cont/spec.cont
        orig = spec.copy()
        orig.flux /= cratio
        orig.err /= cratio
        orig.cont *= cratio    
        # Make the diagnostic figure
        annotlabels = np.char.array(fitparams).lower()
        annotlabels[rvind] = 'vrel'
        specfigure(figfile,specm,fmodel,out,original=orig,verbose=verbose,ispayne=True,annotlabels=annotlabels)

    # How long did this take
    if verbose is True: print('dt = %5.2f sec.' % (time.time()-t0))
    
    return out, fmodel, specm


def multifit_lsq_payne(speclist,modlist,fitparams=None,fixparams={},initpar=None,verbose=False):
    """
    Least Squares fitting with forward modeling of multiple spectra simultaneously.
    
    Parameters
    ----------
    speclist : Spec1D object
         List of the observed spectra to match.
    modlist : list of Doppler Payne models
         A list of the prepared Doppler Payne models to use, one set for each observed
           spectrum.
    fitparams : list of labels, optional
         List of Payne parameter/label names to fit (excluding RV). Optional.
         The default values are ['TEFF','LOGG','FE_H','ALPHA_H'].
    fixparams : dict, optional
         Dictionary of parameters to hold fixed.
    initpar : numpy array, optional
         Initial estimate for [teff, logg, feh, RV, etc.], optional.
    verbose : bool, optional
         Verbose output of the various steps.  This is False by default.

    Returns
    -------
    out : numpy structured array
         The output structured array of the final derived RVs, stellar parameters and errors.
    bmodel : Spec1D object
         The best-fitting Payne model spectrum (as Spec1D object).

    Example
    -------

    .. code-block:: python

         out, bmodel = multifit_lsq_payne(speclist,modlist,initpar)

    """

    print = utils.getprintfunc() # Get print function to be used locally, allows for easy logging
    
    nspec = len(speclist)

    # Fitting parameters, excluding RV
    if fitparams is None:
        fitparams = ['TEFF','LOGG','FE_H','ALPHA_H']
    # Make sure RV is excluded, that is handled separately
    fitparams = np.char.array(fitparams)
    fitparams = list(fitparams[np.char.array(fitparams).upper().find('RV')==-1])
    nfitparams = len(fitparams)
    nfixparams = len(fixparams)
    
    # Get initial estimates
    npar = nfitparams+nspec

    # Initial estimates
    if initpar is None:
        initpar1 = make_payne_initlabels(fitparams)
        initpar = np.hstack((initpar1,np.zeros(nspec,float)))
        
    # Calculate the bounds
    lbounds = np.zeros(npar,float)+1e5
    ubounds = np.zeros(npar,float)-1e5
    labelbounds = make_payne_bounds(fitparams,initpar[0:nfitparams])
    lbounds[0:nfitparams] = labelbounds[0]
    ubounds[0:nfitparams] = labelbounds[1]    
    lbounds[nfitparams:] = -1000
    ubounds[nfitparams:] = 1000    
    bounds = (lbounds, ubounds)

    # Initialize the fitter
    spfitter = payne.PayneMultiSpecFitter(speclist,modlist,fitparams,fixparams=fixparams,verbose=verbose)

    # Put all of the spectra into a large 1D array
    ntotpix = 0
    for s in speclist:
        ntotpix += s.npix*s.norder
    wave = np.zeros(ntotpix)
    flux = np.zeros(ntotpix)
    err = np.zeros(ntotpix)
    cnt = 0
    for i in range(nspec):
        spec1 = speclist[i]
        npx = spec1.npix*spec1.norder
        wave[cnt:cnt+npx] = spec1.wave.T.flatten()
        flux[cnt:cnt+npx] = spec1.flux.T.flatten()
        err[cnt:cnt+npx] = spec1.err.T.flatten()
        cnt += npx

    # We are fitting nfitparams stellar parameters and Nspec relative RVs    
    lspars, lscov = curve_fit(spfitter.model, wave, flux, sigma=err, p0=initpar, bounds=bounds, jac=spfitter.jac)    
  
    # If it hits a boundary then the solution won't chance much compared to initpar
    # setting absolute_sigma=True gives crazy low lsperror values
    lsperror = np.sqrt(np.diag(lscov))

    if verbose is True:
        print('Least Squares labels and RVs:')
        parnames = fitparams.copy()+list('RV'+np.char.array((np.arange(nspec)+1).astype(str)))
        printpars(lspars,lsperror,names=parnames)
    lsmodel = spfitter.model(wave,*lspars)
    lschisq = np.sqrt(np.sum(((flux-lsmodel)/err)**2)/len(lsmodel))
    if verbose is True: print('chisq = %5.2f' % lschisq)

    # Put it into the output structure
    dtype = np.dtype([('pars',float,npar),('parerr',float,npar),('parcov',float,(npar,npar)),('chisq',float)])
    out = np.zeros(1,dtype=dtype)
    out['pars'] = lspars
    out['parerr'] = lsperror
    out['parcov'] = lscov
    out['chisq'] = lschisq
    
    return out, lsmodel


def jointfit_payne(speclist,model=None,fitparams=None,fixparams={},mcmc=False,snrcut=10.0,
                   saveplot=False,verbose=False,outdir=None,nthreads=None,
                   notweak=False,tpoly=False,tpolyorder=3):
    """
    This fits a Payne model to multiple spectra of the same star.

    Parameters
    ----------
    speclist : Spec1D object
         The observed spectrum to match.
    model : Payne model, optional
         Payne model to use.  The default is to load the Payne model
         in the data/ directory and use those.
    fitparams : list of labels, optional
         List of Payne parameter/label names to fit (excluding RV). Optional.
         The default values are ['TEFF','LOGG','FE_H','ALPHA_H'].
    fixparams : dict, optional
         Dictionary of parameters to hold fixed.
    mcmc : bool, optional
         Run Markov Chain Monte Carlo (MCMC) to get improved parameter uncertainties.
         This is only run the individual spectra are being fit.
         This is False by default.
    snrcut : int, optional
         S/N cut to fit individual spectra in the first step.  The default is snrcut=10.
    saveplot : bool, optional
         Save output plots.
    verbose : bool, optional
         Verbose output of the various steps.  This is False by default.
    outdir : str, optional
         The directory for output files.  The default is to use the current directory.
    nthreads : int, optional
         The number of threads to use.  By default the number of threads is not limited.
    notweak : boolean, optional
         Don't tweak the observed continuum using the model.  Default is False.
    tpoly : boolean, optional
         Use a low-order polynomial fit for continuum tweaking.  Default is False,
           it uses Gaussian smoothing instead.
    tpolyorder : int, optional
         Polynomial order to use for continuum tweaking if tpoly=True.
           Default is 3 (cubic).

    Returns
    -------
    sumstr : numpy structured array
         Summary catalog of final best-fit values.
    final : 
         Final best-fit values for each individual spectrum.
    bmodel : List of Spec1D object
         List of best-fitting model spectra.
    specmlist : list of Spec1D object
         List of the observed spectrua with discrepant and outlier pixels masked.

    Example
    -------

    .. code-block:: python

         sumstr, final, bmodel, specmlist = jointfit_payne(speclist)

    """

    print = utils.getprintfunc() # Get print function to be used locally, allows for easy logging
    
    nspec = len(speclist)
    t0 = time.time()

    # Set threads
    if nthreads is not None:
        os.environ["OMP_NUM_THREADS"] = str(nthreads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
        os.environ["MKL_NUM_THREADS"] = str(nthreads)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(nthreads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(nthreads)    
    
    # If list of filenames input, then load them

    # Load the Payne model
    #---------------------
    if model is None: model = payne.load_models()
    
    # Fitting parameters, excluding RV
    if fitparams is None:
        fitparams = ['TEFF','LOGG','FE_H','ALPHA_H']
    if verbose: print('Fitting: '+', '.join(fitparams))
    # Make sure RV is excluded, that is handled separately
    fitparams = np.char.array(fitparams)
    fitparams = list(fitparams[np.char.array(fitparams).upper().find('RV')==-1])
    nfitparams = len(fitparams)
    nfixparams = len(fixparams)

    if verbose: print('Jointly fitting Payne model to '+str(nspec)+' spectra with parameters: '+
                      ', '.join(fitparams)+' and RV for each spectrum')
    
    # Creating catalog of info on each spectrum
    dtlist = [('filename',np.str,300),('snr',float),('vhelio',float),('vrel',float),('vrelerr',float)]
    for k in range(nfitparams):
        name = fitparams[k].lower()
        dtlist += [(name,np.float32),(name+'err',np.float32)]
    dtlist += [('chisq',float),('bc',float)]
    dt = np.dtype(dtlist)
    info = np.zeros(nspec,dtype=dt)
    for n in dt.names: info[n] = np.nan
    for i,s in enumerate(speclist):
        info['filename'][i] = s.filename
        info['snr'][i] = s.snr

    # Make sure some spectra pass the S/N cut
    hisnr, nhisnr = dln.where(info['snr']>snrcut)
    if nhisnr < np.ceil(0.25*nspec):
        snr = np.flip(np.sort(info['snr']))
        snrcut = snr[np.maximum(np.int(np.ceil(0.25*nspec)),np.minimum(4,nspec-1))]
        if verbose is True:
            print('Lowering S/N cut to %5.1f so at least 25%% of the spectra pass the cut' % snrcut)
        
    # Step 1) Loop through each spectrum and run fit()
    if verbose is True: print('Step #1: Fitting the individual spectra')
    specmlist = []
    modlist = []
    for i in range(len(speclist)):
        spec = speclist[i].copy()
        if verbose is True:
            print('Fitting spectrum '+str(i+1))
            print(speclist[i].filename)
        # Only do this for spectra with S/N>10 or 15
        if spec.snr>snrcut:
            # Save the plot, figure the output figure filename
            figfile = None
            if saveplot is True:
                fdir, base, ext = utils.splitfilename(speclist[i].filename)
                figfile = base+'_dopfit.png'
                if outdir is not None: figfile = outdir+'/'+figfile
                if (outdir is None) & (fdir != ''): figfile = fdir+'/'+figfile
            # Fit the spectrum
            fitparams1 = fitparams+['RV']  # make sure to fit the RV
            out, bmodel, specm = fit_payne(spec,fitparams=fitparams1,fixparams=fixparams,
                                           verbose=verbose,mcmc=mcmc,figfile=figfile,
                                           notweak=notweak,tpoly=tpoly,tpolyorder=tpolyorder)
            # Save the "prepared" DopplerPayneModel object, but don't
            # copy the original data (~200MB).
            pmodel = model.copy()  # points to original data
            pmodel.prepare(specm)   # Now prepare the model
            modlist.append(pmodel)
            specmlist.append(specm.copy())
            info['vhelio'][i] = out['vhelio']
            info['vrel'][i] = out['vrel']
            info['vrelerr'][i] = out['vrelerr']
            for k in range(nfitparams):                
                name = fitparams[k].lower()
                info[name][i] = out[name]
                info[name+'err'][i] = out[name+'err']                
            info['chisq'][i] = out['chisq']
            info['bc'][i] = out['bc']
        else:
            if verbose is True:
                print('Skipping: S/N=%6.1f below threshold of %6.1f.  Loading spectrum and preparing models.' % (spec.snr,snrcut))
            # Just get the spectrum
            sp = speclist[i].copy()
            sp = utils.specprep(sp)   # mask and normalize
            # Mask outliers
            sp = utils.maskoutliers(sp)
            specmlist.append(sp)                
            # Save the "prepared" DopplerPayneModel object, but don't
            # copy the original data (~200MB).
            pmodel = model.copy()   # points to original data
            pmodel.prepare(sp)   # Now prepare the model
            modlist.append(pmodel)
            specmlist.append(specm.copy())
            # at least need BC
            if speclist[i].bc is None:
                info['bc'][i] = speclist[i].barycorr()
            else:
                info['bc'][i] = speclist[i].bc                
        if verbose is True: print(' ')

        
    # Step 2) Find weighted mean labels
    if verbose is True: print('Step #2: Getting weighted mean labels')
    gd, ngd = dln.where(np.isfinite(info['chisq']))
    if ngd>0:
        pars = list(np.char.array(fitparams).lower())+['vhelio']
        parerr = list(np.char.array(fitparams).lower()+'err')+['vrelerr']
        wtpars = np.zeros(len(pars),float)
        for i in range(len(pars)):
            p = info[pars[i]][gd]
            perr = info[parerr[i]][gd]            
            gdp,ngdp,bdp,nbdp = dln.where(perr > 0.0,comp=True)
            # Weighted by 1/perr^2
            if ngdp>0:
                if (nbdp>0): perr[bdp] = np.max(perr[gdp])*2
                mnp = dln.wtmean(p,perr)
                wtpars[i] = mnp
            # Unweighted
            else:
                wtpars[i] = np.mean(p)
        if verbose is True:
            print('Initial weighted parameters are:')
            printpars(wtpars,names=np.char.array(pars).upper())
    else:
        wtpars = np.zeros(nfitparams+1,float)
        wtpars = make_payne_initlabels(fitparams+['RV'])
        if verbose is True:
            print('No good fits.  Using these as intial guesses:')
            printpars(wtpars)
            
    # Make initial guesses for all the parameters, nfitparams labels and Nspec relative RVs
    initpar1 = np.zeros(nfitparams+nspec,float)
    initpar1[0:nfitparams] = wtpars[0:nfitparams]
    # the default is to use mean vhelio + BC for all visit spectra
    initpar1[nfitparams:] = wtpars[nfitparams]-info['bc']  # vhelio = vrel + BC
    # Use the Vrel values from the initial fitting if they are accurate enough
    gdinit,ngdinit = dln.where(np.isfinite(info['vrel']) & (info['snr']>5))
    if ngdinit>0:
        initpar1[nfitparams+gdinit] = info['vrel'][gdinit]

    
    # Step 3) Refit all spectra simultaneous fitting stellar parameters and RVs
    if verbose is True:
        print(' ')
        print('Step #3: Fitting all spectra simultaneously')
    out1, fmodels1 = multifit_lsq_payne(specmlist,modlist,fitparams=fitparams,
                                        fixparams=fixparams,initpar=initpar1)
    stelpars1 = out1['pars'][0,0:nfitparams]
    stelparerr1 = out1['parerr'][0,0:nfitparams]
    vrel1 = out1['pars'][0,nfitparams:]
    vrelerr1 = out1['parerr'][0,nfitparams:]
    vhelio1 = vrel1+info['bc']
    medvhelio1 = np.median(vhelio1)
    vscatter1 = dln.mad(vhelio1)
    verr1 = vscatter1/np.sqrt(nspec)
    if verbose is True:
        print('Parameters:')
        printpars(stelpars1,names=fitparams)
        print('Vhelio = %6.2f +/- %5.2f km/s' % (medvhelio1,verr1))
        print('Vscatter =  %6.3f km/s' % vscatter1)
        print('Vhelio values = ',vhelio1)

        
    # Step 4) Tweak continua and remove outlies
    if verbose is True:
        print(' ')
        print('Step #4: Tweaking continuum and masking outliers')
    if notweak is False:
        for i,spm in enumerate(specmlist):
            # Create parameter list that includes RV at the end
            params1 = list(stelpars1)+[vrel1[i]]
            paramnames1 = fitparams+['RV']
            parinput1 = dict(zip(paramnames1,params1))
            model1 = modlist[i]
            labels1 = model1.mklabels(parinput1)
            bestm = model1(labels1)
            # Tweak the continuum normalization
            spm = tweakcontinuum(spm,bestm,usepoly=tpoly,polyorder=tpolyorder)
            # Mask out very discrepant pixels when compared to the best-fit model
            spm = utils.maskdiscrepant(spm,bestm,verbose=verbose)
            specmlist[i] = spm.copy()
    else:
        if verbose: print('Skipping tweaking')
        
    # Step 5) refit all spectra simultaneous fitting stellar parameters and RVs
    if verbose is True:
        print(' ')
        print('Step #5: Re-fitting all spectra simultaneously')

    # Initial guesses for all the parameters, nfitparams stellar parameters and Nspec relative RVs
    initpar2 = out1['pars'][0]
    out2, fmodels2 = multifit_lsq_payne(specmlist,modlist,fitparams=fitparams,
                                        fixparams=fixparams,initpar=initpar2)
    stelpars2 = out2['pars'][0,0:nfitparams]
    stelparerr2 = out2['parerr'][0,0:nfitparams]
    vrel2 = out2['pars'][0,nfitparams:]
    vrelerr2 = out2['parerr'][0,nfitparams:]
    vhelio2 = vrel2+info['bc']
    medvhelio2 = np.median(vhelio2)
    vscatter2 = dln.mad(vhelio2)
    verr2 = vscatter2/np.sqrt(nspec)
    if verbose is True:
        print('Final parameters:')
        printpars(stelpars2,names=fitparams)
        print('Vhelio = %6.2f +/- %5.2f km/s' % (medvhelio2,verr2))
        print('Vscatter =  %6.3f km/s' % vscatter2)
        print('Vhelio values = ',vhelio2)
        
    # Final output structure, one element per spectrum
    final = info.copy()
    for k in range(nfitparams):                
        name = fitparams[k].lower()
        info[name] = stelpars2[k]
        info[name+'err'] = stelparerr2[k]
    final['vrel'] = vrel2
    final['vrelerr'] = vrelerr2
    final['vhelio'] = vhelio2
    bmodel = []
    totchisq = 0.0
    totnpix = 0
    for i in range(nspec):
        # Create parameter list that includes RV at the end
        params1 = list(stelpars2)+[vrel2[i]]
        paramnames1 = fitparams+['RV']
        parinput1 = dict(zip(paramnames1,params1))
        model1 = modlist[i]
        labels1 = model1.mklabels(parinput1)
        m = model1(labels1)
        sp1 = specmlist[i]
        chisq = np.sqrt(np.sum(((sp1.flux-m.flux)/sp1.err)**2)/(sp1.npix*sp1.norder))
        totchisq += np.sum(((sp1.flux-m.flux)/sp1.err)**2)
        totnpix += sp1.npix*sp1.norder
        final['chisq'][i] = chisq
        bmodel.append(m)
    totchisq = np.sqrt(totchisq/totnpix)
    
    # Average values
    sumdtlist = [('medsnr',float),('totsnr',float),('vhelio',float),('vscatter',float),('verr',float)]
    for k in range(nfitparams):
        name = fitparams[k].lower()
        sumdtlist += [(name,np.float32),(name+'err',np.float32)]
    sumdtlist += [('chisq',float)]
    sumdt = np.dtype(sumdtlist)
    sumstr = np.zeros(1,dtype=sumdt)
    sumstr['medsnr'] = np.median(info['snr'])
    sumstr['totsnr'] = np.sqrt(np.sum(info['snr']**2))
    sumstr['vhelio'] = medvhelio2
    sumstr['vscatter'] = vscatter2
    sumstr['verr'] = verr2
    for k in range(nfitparams):
        name = fitparams[k].lower()
        sumstr[name] = stelpars2[k]
        sumstr[name+'err'] = stelparerr2[k]
    sumstr['chisq'] = totchisq

    # Save the best-fitting plots
    if saveplot is True:
        if verbose is True:
            print('')
            print('Making best-fit plots for each spectrum')
        pdfnames = []
        specfiles = [s.filename for s in speclist]
        for i,f in enumerate(specfiles):
            # Figure the output figure filename
            fdir,base,ext = utils.splitfilename(speclist[i].filename)
            figname = base+'_dopjointfit.png'
            if outdir is not None: figname = outdir+'/'+figname
            if (outdir is None) & (fdir != ''): figname = fdir+'/'+figname
            # Make the plot
            spec = speclist[i]
            specm = specmlist[i]
            fmodel = bmodel[i]
            fout = final[i]
            # Apply continuum tweak to original spectrum as well
            orig = spec.copy()
            if orig.normalized is False: orig.normalize()
            cratio = specm.cont/orig.cont
            orig.flux /= cratio
            orig.err /= cratio
            orig.cont *= cratio
            annotlabels = np.char.array(fitparams+['RV']).lower()
            annotlabels[-1] = 'vrel'
            specfigure(figname,specm,fmodel,fout,original=orig,verbose=True,
                       ispayne=True,annotlabels=annotlabels)
            # Make a PDF version that we will combine at the end
            fignamepdf = figname.replace('.png','.pdf')
            specfigure(fignamepdf,specm,fmodel,fout,original=orig,verbose=False,
                       ispayne=True,annotlabels=annotlabels)
            pdfnames.append(fignamepdf)
        # Combine the PDFs into one
        fdir,base,ext = utils.splitfilename(specfiles[0])
        combname = base+'_dopjointfit_comb.pdf'  # png
        if outdir is not None: combname = outdir+'/'+combname
        if (outdir is None) & (fdir != ''): combname = fdir+'/'+combname
        if os.path.exists(combname): os.remove(combname)
        cmd = ['gs','-dBATCH','-dNOPAUSE','-q','-sDEVICE=pdfwrite','-sOutputFile='+combname]
        cmd = cmd+pdfnames
        try:
            out = subprocess.call(cmd,stderr=subprocess.STDOUT,shell=False)
            if verbose: print('Combined plots saved to '+combname)
        except subprocess.CalledProcessError:
            raise Exception("Could not combine PDFs with ghostscript")
        # Delete individual PDFs
        for fp in pdfnames: os.remove(fp)

  
    # How long did this take
    if verbose is True: print('dt = %5.2f sec.' % (time.time()-t0))

    return sumstr, final, bmodel, specmlist


######################### CANNON PROGRAMS ###########################


   
def emcee_lnlike_cannon(theta, x, y, yerr, models, spec):
    """
    This helper function calculates the log likelihood for the MCMC portion of fit().
    
    Parameters
    ----------
    theta : array
      Input parameters [teff, logg, feh, rv].
    x : array
      Array of x-values for y.  Not really used.
    y : array
       Observed flux array.
    yerr : array
        Uncertainties in the observed flux.
    models : list of Cannon models
        List of Cannon models to use
    spec : Spec1D
        The observed spectrum.  Needed to run cannon.model_spectrum().

    Outputs
    -------
    lnlike : float
         The log likelihood value.

    """
    #m = cannon.model_spectrum(models,spec,teff=theta[0],logg=theta[1],feh=theta[2],rv=theta[3])
    m = models(teff=theta[0],logg=theta[1],feh=theta[2],rv=theta[3])    
    inv_sigma2 = 1.0/yerr**2
    return -0.5*(np.sum((y-m.flux.flatten())**2*inv_sigma2))


def emcee_lnprior_cannon(theta, models):
    """
    This helper function calculates the log prior for the MCMC portion of fit().
    It's a flat/uniform prior across the stellar parameter space covered by the
    Cannon models.
    
    Parameters
    ----------
    theta : array
      Input parameters [teff, logg, feh, rv].
    models : list of Cannon models
        List of Cannon models to use

    Outputs
    -------
    lnprior : float
         The log prior value.

    """
    for m in models:
        inside = True
        for i in range(3):
            inside &= (theta[i]>=m.ranges[i,0]) & (theta[i]<=m.ranges[i,1])
        inside &= (np.abs(theta[3]) <= 1000)
        if inside:
            return 0.0
    return -np.inf


def emcee_lnprob_cannon(theta, x, y, yerr, models, spec):
    """
    This helper function calculates the log probability for the MCMC portion of fit().
    
    Parameters
    ----------
    theta : array
      Input parameters [teff, logg, feh, rv].
    x : array
      Array of x-values for y.  Not really used.
    y : array
       Observed flux array.
    yerr : array
        Uncertainties in the observed flux.
    models : list of Cannon models
        List of Cannon models to use
    spec : Spec1D
        The observed spectrum.  Needed to run cannon.model_spectrum().

    Outputs
    -------
    lnprob : float
         The log probability value, which is the sum of the log prior and the
         log likelihood.

    """
    lp = emcee_lnprior_cannon(theta,models)
    if not np.isfinite(lp):
        return -np.inf
    return lp + emcee_lnlike_cannon(theta, x, y, yerr, models, spec)


def fit_xcorrgrid_cannon(spec,models=None,samples=None,verbose=False,maxvel=1000.0):
    """
    Fit spectrum using cross-correlation with models sampled in the parameter space.
    
    Parameters
    ----------
    spec : Spec1D object
         The observed spectrum to match.
    models : list of Cannon models, optional
         A list of Cannon models to use.  The default is to load all of the Cannon
         models in the data/ directory and use those.
    samples : numpy structured array, optional
         Catalog of teff/logg/feh parameters to use when sampling the parameter space.
    verbose : bool, optional
         Verbose output of the various steps.  This is False by default.
    maxvel : float, optional
         The maximum velocity to probe in the cross-correlation.  The default is 1000 km/s.

    Returns
    -------
    out : numpy structured array
         The output structured array of the final derived RVs, stellar parameters and errors.
    bmodel : Spec1D object
         The best-fitting Cannon model spectrum (as Spec1D object).

    Example
    -------

    .. code-block:: python

         out, bmodel = fit_xcorrgrid_cannon(spec)


    """

    print = utils.getprintfunc() # Get print function to be used locally, allows for easy logging
    
    # Check that the samples input has the right columns
    if samples is not None:
        for n in ['teff','logg','feh']:
            try:
                dum = samples[n]
            except:
                raise ValueError(n+' not found in input SAMPLES')
            
    # Step 1: Prepare the spectrum
    #-----------------------------
    # normalize and mask spectrum
    spec = utils.specprep(spec)

    # Step 2: Load and prepare the Cannon models
    #-------------------------------------------
    if models is None:
        models = cannon.models.copy()
        models.prepare(spec)
    
    # Step 3: put on logarithmic wavelength grid
    #-------------------------------------------
    wavelog = utils.make_logwave_scale(spec.wave,vel=0.0)  # get new wavelength solution
    obs = spec.interp(wavelog)
    # The LSF information will not be correct if using Gauss-Hermite, it uses a Gaussian approximation
    # it's okay because the "models" are prepared for the original spectra (above)
    
    # Step 4: get initial RV using cross-correlation with rough sampling of Teff/logg parameter space
    #------------------------------------------------------------------------------------------------
    dwlog = np.median(dln.slope(np.log10(wavelog)))
    # vrel = ( 10**(xshift*dwlog)-1 )*cspeed
    maxlag = np.int(np.ceil(np.log10(1+maxvel/cspeed)/dwlog))
    maxlag = np.maximum(maxlag,50)
    if samples is None:
        #teff = [3500.0, 4000.0, 5000.0, 6000.0, 7500.0, 9000.0, 15000.0, 25000.0, 40000.0,  3500.0, 4300.0, 4700.0, 5200.0]
        #logg = [4.8, 4.8, 4.6, 4.4, 4.0, 4.0, 4.0, 4.0, 8.0,  0.5, 1.0, 2.0, 3.0]
        # temporarily remove WD until that's fixed        
        teff = [3500.0, 4000.0, 5000.0, 6000.0, 7500.0, 15000.0, 25000.0, 40000.0,  3500.0, 4300.0, 4700.0, 5200.0]
        logg = [4.8, 4.8, 4.6, 4.4, 4.0, 4.0, 4.0, 4.0,  0.5, 1.0, 2.0, 3.0]        
        feh = -0.5
        dt = np.dtype([('teff',float),('logg',float),('feh',float)])
        samples = np.zeros(len(teff),dtype=dt)
        samples['teff'][:] = teff
        samples['logg'][:] = logg
        samples['feh'][:] = feh
    outdtype = np.dtype([('xshift',np.float32),('vrel',np.float32),('vrelerr',np.float32),('ccpeak',np.float32),('ccpfwhm',np.float32),
                         ('chisq',np.float32),('teff',np.float32),('logg',np.float32),('feh',np.float32)])
    outstr = np.zeros(len(teff),dtype=outdtype)
    if verbose is True: print('TEFF    LOGG     FEH    VREL   CCPEAK    CHISQ')
    for i in range(len(samples)):
        m = models([samples['teff'][i],samples['logg'][i],samples['feh'][i]],rv=0,wave=wavelog)
        outstr1 = specxcorr(m.wave,m.flux,obs.flux,obs.err,maxlag)
        if verbose is True:
            print('%7.2f  %5.2f  %5.2f  %5.2f  %5.2f  %5.2f' % (teff[i],logg[i],feh,outstr1['vrel'][0],outstr1['ccpeak'][0],outstr1['chisq'][0]))
        for n in ['xshift','vrel','vrelerr','ccpeak','ccpfwhm','chisq']: outstr[n][i] = outstr1[n]
        outstr['teff'][i] = teff[i]
        outstr['logg'][i] = logg[i]
        outstr['feh'][i] = feh
    # Get best fit
    bestind = np.argmin(outstr['chisq'])    
    beststr = outstr[bestind]
    bestmodel = models(teff=beststr['teff'],logg=beststr['logg'],feh=beststr['feh'],rv=beststr['vrel'])
    
    if verbose is True:
        print('Initial RV fit:')
        printpars([beststr['teff'],beststr['logg'],beststr['feh'],beststr['vrel']],[None,None,None,beststr['vrelerr']])

    return beststr, bestmodel


def fit_lsq_cannon(spec,models=None,initpar=None,verbose=False):
    """
    Least Squares fitting with forward modeling of the spectrum.
    
    Parameters
    ----------
    spec : Spec1D object
         The observed spectrum to match.
    models : list of Cannon models, optional
         A list of Cannon models to use.  The default is to load all of the Cannon
         models in the data/ directory and use those.
    initpar : numpy array, optional
         Initial estimate for [teff, logg, feh, RV], optional.
    verbose : bool, optional
         Verbose output of the various steps.  This is False by default.

    Returns
    -------
    out : numpy structured array
         The output structured array of the final derived RVs, stellar parameters and errors.
    bmodel : Spec1D object
         The best-fitting Cannon model spectrum (as Spec1D object).

    Example
    -------

    .. code-block:: python

         out, bmodel = fit_lsq_cannon(spec)

    """

    print = utils.getprintfunc() # Get print function to be used locally, allows for easy logging
    
    # Prepare the spectrum
    #-----------------------------
    # normalize and mask spectrum
    if spec.normalized is False:
        spec = utils.specprep(spec)

    # Load and prepare the Cannon models
    #-------------------------------------------
    if models is None:
        models = cannon.models.copy()
        models.prepare(spec)
    
    # Get initial estimates
    if initpar is None:
        initpar = np.array([6000.0, 2.5, -0.5, 0.0])
    initpar = np.array(initpar).flatten()
    
    # Calculate the bounds
    lbounds = np.zeros(4,float)+1e5
    ubounds = np.zeros(4,float)-1e5
    for p in models:
        lbounds[0:3] = np.minimum(lbounds[0:3],np.min(p.ranges,axis=1))
        ubounds[0:3] = np.maximum(ubounds[0:3],np.max(p.ranges,axis=1))
    lbounds[3] = -1000
    ubounds[3] = 1000    
    bounds = (lbounds, ubounds)

    # Make sure RV is in the boundaries
    initpar[3] = dln.limit(initpar[3],-999,999)
    
    # function to use with curve_fit
    def spec_interp(x,teff,logg,feh,rv):
        """ This returns the interpolated model for a given spectrum."""
        # The "models" and "spec" must already exist outside of this function
        m = models(teff=teff,logg=logg,feh=feh,rv=rv)
        if m is None:
            return np.zeros(spec.flux.shape,float).flatten()+1e30
        return m.flux.flatten()
    
    # Use curve_fit
    lspars, lscov = curve_fit(spec_interp, spec.wave.flatten(), spec.flux.flatten(), sigma=spec.err.flatten(),
                              p0=initpar, bounds=bounds)
    # If it hits a boundary then the solution won't change much compared to initpar
    # setting absolute_sigma=True gives crazy low lsperror values
    lsperror = np.sqrt(np.diag(lscov))
    
    if verbose is True:
        print('Least Squares RV and stellar parameters:')
        printpars(lspars)
    lsmodel = models(teff=lspars[0],logg=lspars[1],feh=lspars[2],rv=lspars[3])
    lschisq = np.sqrt(np.sum(((spec.flux-lsmodel.flux)/spec.err)**2)/(spec.npix*spec.norder))
    if verbose is True: print('chisq = %5.2f' % lschisq)

    # Put it into the output structure
    dtype = np.dtype([('pars',float,4),('parerr',float,4),('parcov',float,(4,4)),('chisq',float)])
    out = np.zeros(1,dtype=dtype)
    out['pars'] = lspars
    out['parerr'] = lsperror
    out['parcov'] = lscov
    out['chisq'] = lschisq
    
    return out, lsmodel


def fit_mcmc_cannon(spec,models=None,initpar=None,steps=100,cornername=None,verbose=False):
    """
    Fit the spectrum with MCMC.
    
    Parameters
    ----------
    spec : Spec1D object
         The observed spectrum to match.
    models : list of Cannon models, optional
         A list of Cannon models to use.  The default is to load all of the Cannon
         models in the data/ directory and use those.
    initpar : numpy array, optional
         Initial estimate for [teff, logg, feh, RV], optional.
    steps : int, optional
         Number of steps to use.  Default is 100.
    cornername : string, optional
         Output filename for the corner plot.  If a corner plot is requested, then the
         minimum number of steps used is 500.
    verbose : bool, optional
         Verbose output of the various steps.  This is False by default.

    Returns
    -------
    out : numpy structured array
         The output structured array of the final derived RVs, stellar parameters and errors.
    bmodel : Spec1D object
         The best-fitting Cannon model spectrum (as Spec1D object).

    Example
    -------

    .. code-block:: python

         out, bmodel = fit_mcmc(spec)

    """

    print = utils.getprintfunc() # Get print function to be used locally, allows for easy logging
    
    # Prepare the spectrum
    #-----------------------------
    # normalize and mask spectrum
    if spec.normalized is False:
        spec = utils.specprep(spec)

    # Load and prepare the Cannon models
    #-------------------------------------------
    if models is None:
        models = cannon.models.copy()
        models.prepare(spec)

    # Initial estimates
    if initpar is None:
        initpar = [6000.0, 2.5, -0.5, 0.0]

    # Set up the MCMC sampler
    ndim, nwalkers = 4, 20
    delta = [initpar[0]*0.1, 0.1, 0.1, 0.2]
    pos = [initpar + delta*np.random.randn(ndim) for i in range(nwalkers)]
        
    sampler = emcee.EnsembleSampler(nwalkers, ndim, emcee_lnprob_cannon,
                                    args=(spec.wave.flatten(), spec.flux.flatten(), spec.err.flatten(), models, spec))

    if cornername is not None: steps=np.maximum(steps,500)  # at least 500 steps
    out = sampler.run_mcmc(pos, steps)

    samples = sampler.chain[:, np.int(steps/2):, :].reshape((-1, ndim))

    # Get the median and stddev values
    pars = np.zeros(ndim,float)
    parerr = np.zeros(ndim,float)
    if verbose is True: print('MCMC values:')
    names = ['Teff','logg','[Fe/H]','Vrel']
    for i in range(ndim):
        t=np.percentile(samples[:,i],[16,50,84])
        pars[i] = t[1]
        parerr[i] = (t[2]-t[0])*0.5
    if verbose is True: printpars(pars,parerr)
        
    # The maximum likelihood parameters
    bestind = np.unravel_index(np.argmax(sampler.lnprobability),sampler.lnprobability.shape)
    pars_ml = sampler.chain[bestind[0],bestind[1],:]

    mcmodel = models(teff=pars[0],logg=pars[1],feh=pars[2],rv=pars[3])
    mcchisq = np.sqrt(np.sum(((spec.flux-mcmodel.flux)/spec.err)**2)/(spec.npix*spec.norder))

    # Put it into the output structure
    dtype = np.dtype([('pars',float,4),('pars_ml',float,4),('parerr',float,4),('chisq',float)])
    out = np.zeros(1,dtype=dtype)
    out['pars'] = pars
    out['pars_ml'] = pars_ml    
    out['parerr'] = parerr
    out['chisq'] = mcchisq
    
    # Corner plot
    if cornername is not None:
        backend = matplotlib.rcParams['backend']        
        matplotlib.use('Agg')
        fig = corner.corner(samples, labels=["T$_eff$", "$\log{g}$", "[Fe/H]", "Vrel"], truths=pars)
        plt.savefig(cornername)
        plt.close(fig)
        print('Corner plot saved to '+cornername)
        matplotlib.use(backend)  # back to the original backend
        
    return out,mcmodel


def fit_cannon(spectrum,models=None,estimates=None,verbose=False,mcmc=False,figfile=None,
               cornername=None,retpmodels=False,nthreads=None,notweak=False,tpoly=False,
               tpolyorder=3):
    """
    Fit the spectrum.  Find the best RV and stellar parameters using the Cannon models.

    Parameters
    ----------
    spectrum : Spec1D object
         The observed spectrum to match.
    models : list of Cannon models, optional
         A list of Cannon models to use.  The default is to load all of the Cannon
         models in the data/ directory and use those.
    estimates : dict, optional
         Dictionary of initial guesses.
    verbose : bool, optional
         Verbose output of the various steps.  This is False by default.
    mcmc : bool, optional
         Run Markov Chain Monte Carlo (MCMC) to get improved parameter uncertainties.
         This is False by default.
    figfile : string, optional
         The filename for a diagnostic plot showing the observed spectrum, model
         spectrum and the best-fit parameters.
    cornername : string, optional
         The filename for a "corner" plot showing the posterior distributions from
         the MCMC run.
    retpmodels : bool, optional
         Return the prepared moels.
    nthreads : int, optional
         The number of threads to use.  By default the number of threads is not limited.
    notweak : boolean, optional
         Don't tweak the observed continuum using the model.  Default is False.
    tpoly : boolean, optional
         Use a low-order polynomial fit for continuum tweaking.  Default is False,
           it uses Gaussian smoothing instead.
    tpolyorder : int, optional
         Polynomial order to use for continuum tweaking if tpoly=True.
           Default is 3 (cubic).

    Returns
    -------
    out : numpy structured array
         The output structured array of the final derived RVs, stellar parameters and errors.
    model : Spec1D object
         The best-fitting Cannon model spectrum (as Spec1D object).
    specm : Spec1D object
         The observed spectrum with discrepant and outlier pixels masked.
    pmodels : DopplerCannonModelSet
         The prepared Doppler Cannon models, only if retpmodels=True.

    Example
    -------

    .. code-block:: python

         out, model = fit_cannon(spec)

    """

    print = utils.getprintfunc() # Get print function to be used locally, allows for easy logging
    
    # Turn off the Cannon's info messages
    tclogger = logging.getLogger('thecannon.utils')
    tclogger.disabled = True

    # Set threads
    if nthreads is not None:
        os.environ["OMP_NUM_THREADS"] = str(nthreads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
        os.environ["MKL_NUM_THREADS"] = str(nthreads)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(nthreads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(nthreads)    
    
    t0 = time.time()

    if verbose: print('Fitting Cannon model to spectrum with parameters: '+', '.join(['Teff','logg','[Fe/H]','RV']))        
    
    # Make internal copy
    spec = spectrum.copy()
    
    # Step 1: Prepare the spectrum
    #-----------------------------
    # Normalize and mask the spectrum
    spec = utils.specprep(spec)         
    # Mask out any large positive outliers, e.g. badly subtracted sky lines
    specm = utils.maskoutliers(spec,verbose=verbose)
    
    # Step 2: Load and prepare the Cannon models
    #-------------------------------------------
    if models is None: models = cannon.models.copy()
    models.prepare(specm)
    ##  NOT interpolated onto the observed wavelength scale

    # Step 3: Get initial RV using cross-correlation with rough sampling of Teff/logg parameter space
    #------------------------------------------------------------------------------------------------
    doxcorr,beststr = True,None
    if estimates is not None:
        uest = {k.upper():estimates[k] for k in estimates.keys()}
        ukeys = uest.keys()
        if 'TEFF' in ukeys and 'LOGG' in ukeys and 'FE_H' in ukeys and 'RV' in ukeys:
            doxcorr = False
    if doxcorr:
        beststr, xmodel = fit_xcorrgrid_cannon(specm,models,verbose=verbose,maxvel=1000.0)  
    # Replace values with estimates
    if estimates is not None:
        if verbose: print('Using input estimates: '+str(estimates))
        if beststr is None:
            beststr = {'teff':uest['TEFF'],'logg':uest['LOGG'],'feh':uest['FE_H'],'vrel':uest['RV']}
        else:
            for k in ['TEFF','LOGG','FE_H','RV']:
                if k in ukeys:
                    if k=='FE_H':
                        beststr['feh'] = uest[k]
                    else:
                        beststr[k.lower()] = uest[k]
    
    # Step 4: Get better Cannon stellar parameters using initial RV
    #--------------------------------------------------------------
    # put observed spectrum on rest wavelength scale
    # get cannon model for "best" teff/logg/feh values
    # run cannon.test() on the spectrum and variances
    # just shift the observed wavelengths to rest, do NOT interpolate the spectrum
    #restwave = obs.wave*(1-beststr['vrel']/cspeed)
    restwave = specm.wave*(1-beststr['vrel']/cspeed)    
    bestmodel = models.get_best_model([beststr['teff'],beststr['logg'],beststr['feh']])
    bestmodelinterp = bestmodel.interp(restwave)
    labels0, cov0, meta0 = bestmodelinterp.test(specm)
    
    # Make sure the labels are within the ranges
    labels0 = labels0.flatten()
    for i in range(3): labels0[i]=dln.limit(labels0[i],bestmodelinterp.ranges[i,0],bestmodelinterp.ranges[i,1])
    bestmodelspec0 = bestmodelinterp(labels0)
    if verbose is True:
        print('Initial Cannon stellar parameters using initial RV')
        printpars(labels0) 

    # Tweak the continuum normalization
    if notweak is False:
        specm = tweakcontinuum(specm,bestmodelspec0,usepoly=tpoly,polyorder=tpolyorder)        
        # Mask out very discrepant pixels when compared to the best-fit model
        specm = utils.maskdiscrepant(specm,bestmodelspec0,verbose=verbose)  
    
    # Refit the Cannon
    labels, cov, meta = bestmodelinterp.test(specm)
    # Make sure the labels are within the ranges
    labels = labels.flatten()
    for i in range(3): labels[i]=dln.limit(labels[i],bestmodelinterp.ranges[i,0],bestmodelinterp.ranges[i,1])
    bestmodelspec = bestmodelinterp(labels)
    if verbose is True:
        print('Initial Cannon stellar parameters using initial RV and Tweaking the normalization')
        printpars(labels)
    
    # Step 5: Improved RV using better Cannon template
    #-------------------------------------------------
    wavelog = utils.make_logwave_scale(specm.wave,vel=0.0)  # get new wavelength solution
    obs = specm.interp(wavelog)
    m = models.get_best_model(labels).interp(wavelog)(labels,rv=0)
    dwlog = np.median(dln.slope(np.log10(wavelog)))
    # vrel = ( 10**(xshift*dwlog)-1 )*cspeed
    maxlag = np.int(np.ceil(np.log10(1+1000.0/cspeed)/dwlog))
    maxlag = np.maximum(maxlag,50)
    outstr2 = specxcorr(m.wave,m.flux,obs.flux,obs.err,maxlag)
    outdtype = np.dtype([('xshift',np.float32),('vrel',np.float32),('vrelerr',np.float32),('ccpeak',np.float32),('ccpfwhm',np.float32),
                         ('chisq',np.float32),('teff',np.float32),('logg',np.float32),('feh',np.float32)])
    beststr2= np.zeros(1,dtype=outdtype)
    for n in ['xshift','vrel','vrelerr','ccpeak','ccpfwhm','chisq']: beststr2[n] = outstr2[n]
    beststr2['teff'] = labels[0]
    beststr2['logg'] = labels[1]
    beststr2['feh'] = labels[2]

    
    # Step 6: Improved Cannon stellar parameters
    #-------------------------------------------
    restwave = specm.wave*(1-beststr['vrel']/cspeed)    
    bestmodel = models.get_best_model([beststr2['teff'],beststr2['logg'],beststr2['feh']])
    bestmodelinterp = bestmodel.interp(restwave)
    labels2, cov2, meta2 = bestmodelinterp.test(specm)
    # Make sure the labels are within the ranges
    labels2 = labels2.flatten()
    for i in range(3): labels2[i]=dln.limit(labels2[i],bestmodelinterp.ranges[i,0],bestmodelinterp.ranges[i,1])
    bestmodelspec2 = bestmodelinterp(labels2)
    if verbose is True:
        print('Improved RV and Cannon stellar parameters:')
        printpars(np.concatenate((labels2,beststr2['vrel'])),[None,None,None,beststr2['vrelerr']])
    
    # Step 7: Least Squares fitting with forward modeling
    #----------------------------------------------------
    # Get best model so far
    m = models(teff=beststr2['teff'],logg=beststr2['logg'],feh=beststr2['feh'],rv=beststr2['vrel'])
    # Tweak the continuum
    if notweak is False:
        specm = tweakcontinuum(specm,m,usepoly=tpoly,polyorder=tpolyorder)

    # Get initial estimates
    initpar = [beststr2['teff'],beststr2['logg'],beststr2['feh'],beststr2['vrel']]
    initpar = np.array(initpar).flatten()
    lsout, lsmodel = fit_lsq_cannon(specm,models,initpar=initpar,verbose=verbose)
    lspars = lsout['pars'][0]
    lsperror = lsout['parerr'][0]    
    
    # Step 8: Run fine grid in RV, forward modeling
    #----------------------------------------------
    maxv = np.maximum(beststr2['vrel'][0],20.0)
    vel = dln.scale_vector(np.arange(30),lspars[3]-maxv,lspars[3]+maxv)
    chisq = np.zeros(len(vel))
    for i,v in enumerate(vel):
        m = models(teff=lspars[0],logg=lspars[1],feh=lspars[2],rv=v)
        chisq[i] = np.sqrt(np.sum(((specm.flux-m.flux)/specm.err)**2)/(specm.npix*specm.norder))
    vel2 = dln.scale_vector(np.arange(300),lspars[3]-maxv,lspars[3]+maxv)
    chisq2 = dln.interp(vel,chisq,vel2)
    bestind = np.argmin(chisq2)
    finerv = vel2[bestind]
    finechisq = chisq2[bestind]
    if verbose is True:
        print('Fine grid best RV = %5.2f km/s' % finerv)
        print('chisq = %5.2f' % finechisq)
    
    # Final parameters and uncertainties (so far)
    fpars = lspars
    fperror = lsperror
    fpars[3] = finerv
    fchisq = finechisq
    fmodel = models(teff=lspars[0],logg=lspars[1],feh=lspars[2],rv=finerv)                            
    
    # Step 9: MCMC
    #--------------
    if (mcmc is True) | (cornername is not None):
        mcout, mcmodel = fit_mcmc_cannon(specm,models,fpars,verbose=verbose,cornername=cornername)
        # Use these parameters
        fpars = mcout['pars'][0]
        fperror = mcout['parerr'][0]
        fchisq = mcout['chisq'][0]
        fmodel = mcmodel
        
    # Construct the output
    #---------------------
    if specm.bc is None:
        bc = specm.barycorr()
    else:
        bc = specm.bc
    # Check if we should apply the barycentric correction
    #  some spectra already this applied to the wavelength solution
    nobc = False
    if hasattr(spec,'nobc'):
        nobc = spec.nobc
    # Apply barycentric correction
    if nobc is False:
        vhelio = fpars[3] + bc
    else:
        vhelio = fpars[3]
        if verbose is True:
            print('NOT applying Barycentric Correction')
    if verbose is True:
        print('Final parameters:')
        printpars(fpars[0:3],fperror[0:3])
        print('Vhelio = %6.2f +/- %5.2f km/s' % (vhelio,fperror[3]))
        print('BC = %5.2f km/s' % bc)
        print('chisq = %5.2f' % fchisq)
    dtype = np.dtype([('vhelio',np.float32),('vrel',np.float32),('vrelerr',np.float32),
                      ('teff',np.float32),('tefferr',np.float32),('logg',np.float32),('loggerr',np.float32),
                      ('feh',np.float32),('feherr',np.float32),('chisq',np.float32),('bc',np.float32)])
    out = np.zeros(1,dtype=dtype)
    out['vhelio'] = vhelio
    out['vrel'] = fpars[3]
    out['vrelerr'] = fperror[3]    
    out['teff'] = fpars[0]
    out['tefferr'] = fperror[0]    
    out['logg'] = fpars[1]
    out['loggerr'] = fperror[1]        
    out['feh'] = fpars[2]
    out['feherr'] = fperror[2]    
    out['chisq'] = fchisq
    out['bc'] = bc

    # Make diagnostic figure
    if figfile is not None:
        # Apply continuum tweak to original spectrum as well
        cratio = specm.cont/spec.cont
        orig = spec.copy()
        orig.flux /= cratio
        orig.err /= cratio
        orig.cont *= cratio    
        # Make the diagnostic figure
        specfigure(figfile,specm,fmodel,out,original=orig,verbose=verbose)

    # How long did this take
    if verbose is True: print('dt = %5.2f sec.' % (time.time()-t0))
    
    # Return the prepared models
    if retpmodels is True:
        return out, fmodel, specm, models
    
    return out, fmodel, specm
    

def multifit_lsq_cannon(speclist,modlist,initpar=None,verbose=False):
    """
    Least Squares fitting with forward modeling of multiple spectra simultaneously.
    
    Parameters
    ----------
    speclist : Spec1D object
         List of the observed spectra to match.
    modlist : list of Doppler Cannon models
         A list of the prepare Doppler Cannon models to use, one set for each observed
           spectrum.
    initpar : numpy array, optional
         Initial estimate for [teff, logg, feh, RV], optional.
    verbose : bool, optional
         Verbose output of the various steps.  This is False by default.

    Returns
    -------
    out : numpy structured array
         The output structured array of the final derived RVs, stellar parameters and errors.
    bmodel : Spec1D object
         The best-fitting Cannon model spectrum (as Spec1D object).

    Example
    -------

    .. code-block:: python

         out, bmodel = multifit_lsq(speclist,modlist,initpar)

    """

    print = utils.getprintfunc() # Get print function to be used locally, allows for easy logging
    
    nspec = len(speclist)
    
    ## Prepare the spectrum
    ##-----------------------------
    ## normalize and mask spectrum
    #if spec.normalized is False: spec.normalize()
    #if spec.mask is not None:
    #    # Set errors to high value, leave flux alone
    #    spec.err[spec.mask] = 1e30

    ## Load and prepare the Cannon models
    ##-------------------------------------------
    #if models is None: models = cannon.models.prepare(spec)
    
    # Get initial estimates
    npar = 3+nspec
    if initpar is None:
        initpar = np.zeros(npar,float)
        initpar[0:3] = np.array([6000.0, 2.5, -0.5])
    
    # Calculate the bounds
    lbounds = np.zeros(npar,float)+1e5
    ubounds = np.zeros(npar,float)-1e5
    for p in modlist[0]:
        lbounds[0:3] = np.minimum(lbounds[0:3],np.min(p.ranges,axis=1))
        ubounds[0:3] = np.maximum(ubounds[0:3],np.max(p.ranges,axis=1))
    lbounds[3:] = -1000
    ubounds[3:] = 1000    
    bounds = (lbounds, ubounds)

    # Limit the initial guess RVs
    initpar[3:] = np.maximum(np.minimum(initpar[3:],950),-950)
    
    # function to use with curve_fit
    def multispec_interp(x,*argv):
        """ This returns the interpolated model for a given spectrum."""
        # The "models" and "spec" must already exist outside of this function
        #print(argv)
        teff = argv[0]
        logg = argv[1]
        feh = argv[2]
        vrel = argv[3:]
        npix = len(x)
        nspec = len(vrel)
        flux = np.zeros(npix,float)
        cnt = 0
        for i in range(nspec):
            npx = speclist[i].npix*speclist[i].norder
            m = modlist[i]([teff,logg,feh],rv=vrel[i])
            if m is not None:
                flux[cnt:cnt+npx] = m.flux.T.flatten()
            else:
                flux[cnt:cnt+npx] = 1e30
            cnt += npx
        return flux

    def multispec_interp_jac(x,*argv):
        """ Compute the Jacobian matrix (an m-by-n matrix, where element (i, j)
            is the partial derivative of f[i] with respect to x[j]). """
        # We only have to recompute the full model if teff/logg/feh are being modified
        # otherwise we just modify one spectrum's model
        #print('jac')
        #print(argv)
        relstep = 0.02
        npix = len(x)
        npar = len(argv)
        teff = argv[0]
        logg = argv[1]
        feh  = argv[2]
        vrel = argv[3:]
        # Initialize jacobian matrix
        jac = np.zeros((npix,npar),float)
        # Model at current values
        f0 = multispec_interp(x,*argv)
        # Compute full models for teff/logg/feh
        for i in range(3):
            pars = np.array(copy.deepcopy(argv))
            step = relstep*pars[i]
            pars[i] += step
            f1 = multispec_interp(x,*pars)
            # Hit an edge, try the negative value instead
            nbd = np.sum(f1>1000)
            if nbd>1000:
                pars = np.array(copy.deepcopy(argv))
                step = -relstep*pars[i]
                pars[i] += step
                f1 = multispec_interp(x,*pars)
            jac[:,i] = (f1-f0)/step
        # Compute model for single spectra
        nspec = len(speclist)
        cnt = 0
        for i in range(nspec):
            vrel1 = vrel[i]
            step = 1.0
            vrel1 += step
            npx = speclist[i].npix*speclist[i].norder
            m = modlist[i]([teff,logg,feh],rv=vrel1)
            if m is not None:
                jac[cnt:cnt+npx,i] = (m.flux.T.flatten()-f0[cnt:cnt+npx])/step
            else:
                jac[cnt:cnt+npx,i] = 1e30
            cnt += npx
                
        return jac

        
    # We are fitting 3 stellar parameters and Nspec relative RVs

    # Put all of the spectra into a large 1D array
    ntotpix = 0
    for s in speclist:
        ntotpix += s.npix*s.norder
    wave = np.zeros(ntotpix)
    flux = np.zeros(ntotpix)
    err = np.zeros(ntotpix)
    cnt = 0
    for i in range(nspec):
        sp = speclist[i]
        npx = sp.npix*sp.norder
        wave[cnt:cnt+npx] = sp.wave.T.flatten()
        flux[cnt:cnt+npx] = sp.flux.T.flatten()
        err[cnt:cnt+npx] = sp.err.T.flatten()
        cnt += npx
        
    # Use curve_fit
    diff_step = np.zeros(npar,float)
    diff_step[:] = 0.02
    lspars, lscov = curve_fit(multispec_interp, wave, flux, sigma=err, p0=initpar, bounds=bounds, jac=multispec_interp_jac)
    #lspars, lscov = curve_fit(multispec_interp, wave, flux, sigma=err, p0=initpar, bounds=bounds, diff_step=diff_step)    
    #lspars, lscov = curve_fit(multispec_interp, wave, flux, sigma=err, p0=initpar, bounds=bounds)    
    # If it hits a boundary then the solution won't chance much compared to initpar
    # setting absolute_sigma=True gives crazy low lsperror values
    lsperror = np.sqrt(np.diag(lscov))

    if verbose is True:
        print('Least Squares RV and stellar parameters:')
        printpars(lspars)
    lsmodel = multispec_interp(wave,*lspars)
    lschisq = np.sqrt(np.sum(((flux-lsmodel)/err)**2)/len(lsmodel))
    if verbose is True: print('chisq = %5.2f' % lschisq)

    # Put it into the output structure
    dtype = np.dtype([('pars',float,npar),('parerr',float,npar),('parcov',float,(npar,npar)),('chisq',float)])
    out = np.zeros(1,dtype=dtype)
    out['pars'] = lspars
    out['parerr'] = lsperror
    out['parcov'] = lscov
    out['chisq'] = lschisq
    
    return out, lsmodel


def jointfit_cannon(speclist,models=None,mcmc=False,snrcut=10.0,saveplot=False,verbose=False,
                    outdir=None,nthreads=None,notweak=False,tpoly=False,tpolyorder=3):
    """
    This fits a Cannon model to multiple spectra of the same star.

    Parameters
    ----------
    speclist : Spec1D object
         The observed spectrum to match.
    models : list of Cannon models, optional
         A list of Cannon models to use.  The default is to load all of the Cannon
         models in the data/ directory and use those.
    mcmc : bool, optional
         Run Markov Chain Monte Carlo (MCMC) to get improved parameter uncertainties.
         This is only run the individual spectra are being fit.
         This is False by default.
    snrcut : int, optional
         S/N cut to fit individual spectra in the first step.  The default is snrcut=10.
    saveplot : bool, optional
         Save output plots.
    verbose : bool, optional
         Verbose output of the various steps.  This is False by default.
    outdir : str, optional
         The directory for output files.  The default is to use the current directory.
    nthreads : int, optional
         The number of threads to use.  By default the number of threads is not limited.
    notweak : boolean, optional
         Don't tweak the observed continuum using the model.  Default is False.
    tpoly : boolean, optional
         Use a low-order polynomial fit for continuum tweaking.  Default is False,
           it uses Gaussian smoothing instead.
    tpolyorder : int, optional
         Polynomial order to use for continuum tweaking if tpoly=True.
           Default is 3 (cubic).

    Returns
    -------
    sumstr : numpy structured array
         Summary catalog of final best-fit values.
    final : 
         Final best-fit values for each individual spectrum.
    bmodel : List of Spec1D object
         List of best-fitting model spectra.
    specmlist : list of Spec1D object
         List of the observed spectrua with discrepant and outlier pixels masked.

    Example
    -------

    .. code-block:: python

         sumstr, final, bmodel, specmlist = jointfit_cannon(speclist)

    """

    print = utils.getprintfunc() # Get print function to be used locally, allows for easy logging
    
    nspec = len(speclist)
    t0 = time.time()

    # Set threads
    if nthreads is not None:
        os.environ["OMP_NUM_THREADS"] = str(nthreads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(nthreads)
        os.environ["MKL_NUM_THREADS"] = str(nthreads)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(nthreads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(nthreads)    

    if verbose: print('Jointly fitting Cannon model to '+str(nspec)+' spectra with parameters: '+
                      ', '.join(['Teff','logg','[Fe/H]'])+' and RV for each spectrum')
        
    # If list of filenames input, then load them
    
    # Creating catalog of info on each spectrum
    dt = np.dtype([('filename',np.str,300),('snr',float),('vhelio',float),('vrel',float),('vrelerr',float),
                   ('teff',float),('tefferr',float),('logg',float),('loggerr',float),('feh',float),
                   ('feherr',float),('chisq',float),('bc',float)])
    info = np.zeros(nspec,dtype=dt)
    for n in dt.names: info[n] = np.nan
    for i,s in enumerate(speclist):
        info['filename'][i] = s.filename
        info['snr'][i] = s.snr

    # Make sure some spectra pass the S/N cut
    hisnr, nhisnr = dln.where(info['snr']>snrcut)
    if nhisnr < np.ceil(0.25*nspec):
        snr = np.flip(np.sort(info['snr']))
        snrcut = snr[np.maximum(np.int(np.ceil(0.25*nspec)),np.minimum(4,nspec-1))]
        if verbose is True:
            print('Lowering S/N cut to %5.1f so at least 25%% of the spectra pass the cut' % snrcut)
        
    # Step 1) Loop through each spectrum and run fit()
    if verbose is True: print('Step #1: Fitting the individual spectra')
    specmlist = []
    modlist = []
    for i in range(len(speclist)):
        spec = speclist[i].copy()
        if verbose is True:
            print('Fitting spectrum '+str(i+1))
            print(speclist[i].filename)
        # Only do this for spectra with S/N>10 or 15
        if spec.snr>snrcut:
            # Save the plot, figure the output figure filename
            figfile = None
            if saveplot is True:
                fdir, base, ext = utils.splitfilename(speclist[i].filename)
                figfile = base+'_dopfit.png'
                if outdir is not None: figfile = outdir+'/'+figfile
                if (outdir is None) & (fdir != ''): figfile = fdir+'/'+figfile
            # Fit the spectrum    
            out, model, specm, pmodels = fit_cannon(spec,verbose=verbose,mcmc=mcmc,figfile=figfile,
                                                    retpmodels=True,notweak=notweak,tpoly=tpoly,
                                                    tpolyorder=tpolyorder)
            modlist.append(pmodels.copy())
            del pmodels
            specmlist.append(specm.copy())
            del specm
            info['vhelio'][i] = out['vhelio']
            info['vrel'][i] = out['vrel']
            info['vrelerr'][i] = out['vrelerr']    
            info['teff'][i] = out['teff']
            info['tefferr'][i] = out['tefferr']
            info['logg'][i] = out['logg']
            info['loggerr'][i] = out['loggerr']
            info['feh'][i] = out['feh']
            info['feherr'][i] = out['feherr']
            info['chisq'][i] = out['chisq']
            info['bc'][i] = out['bc']
        else:
            if verbose is True:
                print('Skipping: S/N=%6.1f below threshold of %6.1f.  Loading spectrum and preparing models.' % (spec.snr,snrcut))
            modlist.append(cannon.models.copy().prepare(speclist[i]))
            sp = speclist[i].copy()
            sp = utils.specprep(sp)   # mask and normalize
            # Mask outliers
            sp = utils.maskoutliers(sp)
            specmlist.append(sp)
            # at least need BC
            if speclist[i].bc is None:
                info['bc'][i] = speclist[i].barycorr()
            else:
                info['bc'][i] = speclist[i].bc
        if verbose is True: print(' ')

        
    # Step 2) Find weighted stellar parameters
    if verbose is True: print('Step #2: Getting weighted stellar parameters')
    gd, ngd = dln.where(np.isfinite(info['chisq']))
    if ngd>0:
        pars = ['teff','logg','feh','vhelio']
        parerr = ['tefferr','loggerr','feherr','vrelerr']
        wtpars = np.zeros(4,float)
        for i in range(len(pars)):
            p = info[pars[i]][gd]
            perr = info[parerr[i]][gd]
            gdp,ngdp,bdp,nbdp = dln.where(perr > 0.0,comp=True)
            # Weighted by 1/perr^2
            if ngdp>0:
                if (nbdp>0): perr[bdp] = np.max(perr[gdp])*2
                mnp = dln.wtmean(p,perr)
                wtpars[i] = mnp
            # Unweighted
            else:
                wtpars[i] = np.mean(p)
        if verbose is True:
            print('Initial weighted parameters are:')
            printpars(wtpars)
    else:
        wtpars = np.zeros(4,float)
        wtpars[0] = 6000.0
        wtpars[1] = 4.0
        wtpars[2] = -0.5
        wtpars[3] = 0.0
        vscatter0 = 999999.
        if verbose is True:
            print('No good fits.  Using these as intial guesses:')
            printpars(wtpars)
        
    # Make initial guesses for all the parameters, 3 stellar paramters and Nspec relative RVs
    initpar1 = np.zeros(3+nspec,float)
    initpar1[0:3] = wtpars[0:3]
    # the default is to use mean vhelio + BC for all visit spectra
    initpar1[3:] = wtpars[3]-info['bc']  # vhelio = vrel + BC
    # Use the Vrel values from the initial fitting if they are accurate enough
    gdinit,ngdinit = dln.where(np.isfinite(info['vrel']) & (info['snr']>5))
    if ngdinit>0:
        initpar1[gdinit+3] = info['vrel'][gdinit]

    
    # Step 3) Refit all spectra simultaneously fitting stellar parameters and RVs
    if verbose is True:
        print(' ')
        print('Step #3: Fitting all spectra simultaneously')
    out1, fmodels1 = multifit_lsq_cannon(specmlist,modlist,initpar1)
    stelpars1 = out1['pars'][0,0:3]
    stelparerr1 = out1['parerr'][0,0:3]    
    vrel1 = out1['pars'][0,3:]
    vrelerr1 = out1['parerr'][0,3:]
    vhelio1 = vrel1+info['bc']
    medvhelio1 = np.median(vhelio1)
    vscatter1 = dln.mad(vhelio1)
    verr1 = vscatter1/np.sqrt(nspec)
    if verbose is True:
        print('Parameters:')
        printpars(stelpars1)
        print('Vhelio = %6.2f +/- %5.2f km/s' % (medvhelio1,verr1))
        print('Vscatter =  %6.3f km/s' % vscatter1)
        print(vhelio1)

    # Step 4) Tweak continua and remove outliers
    if verbose is True:
        print(' ')
        print('Step #4: Tweaking continuum and masking outliers')
    if notweak is False:
        for i,spm in enumerate(specmlist):
            bestm = modlist[i](stelpars1,rv=vrel1[i])
            # Tweak the continuum normalization
            spm = tweakcontinuum(spm,bestm,usepoly=tpoly,polyorder=tpolyorder)
            # Mask out very discrepant pixels when compared to the best-fit model
            spm = utils.maskdiscrepant(spm,bestm,verbose=verbose)
            specmlist[i] = spm.copy()
    else:
        if verbose: print('Skipping tweaking')
            
            
    # Step 5) Refit all spectra simultaneously fitting stellar parameters and RVs
    if verbose is True:
        print(' ')
        print('Step #5: Re-fitting all spectra simultaneously')

    # Initial guesses for all the parameters, 3 stellar paramters and Nspec relative RVs
    initpar2 = out1['pars'][0]
    out2, fmodels2 = multifit_lsq_cannon(specmlist,modlist,initpar2)
    stelpars2 = out2['pars'][0,0:3]
    stelparerr2 = out2['parerr'][0,0:3]    
    vrel2 = out2['pars'][0,3:]
    vrelerr2 = out2['parerr'][0,3:]
    vhelio2 = vrel2+info['bc']
    medvhelio2 = np.median(vhelio2)
    vscatter2 = dln.mad(vhelio2)
    verr2 = vscatter2/np.sqrt(nspec)
    if verbose is True:
        print('Final parameters:')
        printpars(stelpars2)
        print('Vhelio = %6.2f +/- %5.2f km/s' % (medvhelio2,verr2))
        print('Vscatter =  %6.3f km/s' % vscatter2)
        print(vhelio2)
    
    # Final output structure
    final = info.copy()
    final['teff'] = stelpars2[0]
    final['tefferr'] = stelparerr2[0]
    final['logg'] = stelpars2[1]
    final['loggerr'] = stelparerr2[1]
    final['feh'] = stelpars2[2]
    final['feherr'] = stelparerr2[2]    
    final['vrel'] = vrel2
    final['vrelerr'] = vrelerr2
    final['vhelio'] = vhelio2
    bmodel = []
    totchisq = 0.0
    totnpix = 0
    for i in range(nspec):
        pars1 = [final['teff'][i], final['logg'][i], final['feh'][i]]
        vr1 = final['vrel'][i]
        sp = specmlist[i]
        m = modlist[i](pars1,rv=vr1)
        chisq = np.sqrt(np.sum(((sp.flux-m.flux)/sp.err)**2)/(sp.npix*sp.norder))
        totchisq += np.sum(((sp.flux-m.flux)/sp.err)**2)
        totnpix += sp.npix*sp.norder
        final['chisq'][i] = chisq
        bmodel.append(m)
    totchisq = np.sqrt(totchisq/totnpix)
        
    # Average values
    sumdt = np.dtype([('medsnr',float),('totsnr',float),('vhelio',float),('vscatter',float),('verr',float),
                      ('teff',float),('tefferr',float),('logg',float),('loggerr',float),('feh',float),
                      ('feherr',float),('chisq',float)])
    sumstr = np.zeros(1,dtype=sumdt)
    sumstr['medsnr'] = np.median(info['snr'])
    sumstr['totsnr'] = np.sqrt(np.sum(info['snr']**2))
    sumstr['vhelio'] = medvhelio2
    sumstr['vscatter'] = vscatter2
    sumstr['verr'] = verr2
    sumstr['teff'] = stelpars2[0]
    sumstr['tefferr'] = stelparerr2[0]
    sumstr['logg'] = stelpars2[1]
    sumstr['loggerr'] = stelparerr2[1]
    sumstr['feh'] = stelpars2[2]
    sumstr['feherr'] = stelparerr2[2]
    sumstr['chisq'] = totchisq

    # Save the best-fitting plots
    if saveplot is True:
        if verbose is True:
            print('')
            print('Making best-fit plots for each spectrum')
        pdfnames = []
        specfiles = [s.filename for s in speclist]
        for i,f in enumerate(specfiles):
            # Figure the output figure filename
            fdir,base,ext = utils.splitfilename(speclist[i].filename)
            figname = base+'_dopjointfit.png'
            if outdir is not None: figname = outdir+'/'+figname
            if (outdir is None) & (fdir != ''): figname = fdir+'/'+figname
            # Make the plot
            spec = speclist[i]
            specm = specmlist[i]
            fmodel = bmodel[i]
            fout = final[i]
            # Apply continuum tweak to original spectrum as well
            orig = spec.copy()
            if orig.normalized is False: orig.normalize()
            cratio = specm.cont/orig.cont
            orig.flux /= cratio
            orig.err /= cratio
            orig.cont *= cratio
            specfigure(figname,specm,fmodel,fout,original=orig,verbose=True)
            # Make a PDF version that we will combine at the end
            fignamepdf = figname.replace('.png','.pdf')
            specfigure(fignamepdf,specm,fmodel,fout,original=orig,verbose=False)
            pdfnames.append(fignamepdf)
        # Combine the PDFs into one
        fdir,base,ext = utils.splitfilename(specfiles[0])
        combname = base+'_dopjointfit_comb.pdf'  # png
        if outdir is not None: combname = outdir+'/'+combname
        if (outdir is None) & (fdir != ''): combname = fdir+'/'+combname
        if os.path.exists(combname): os.remove(combname)
        cmd = ['gs','-dBATCH','-dNOPAUSE','-q','-sDEVICE=pdfwrite','-sOutputFile='+combname]
        cmd = cmd+pdfnames
        try:
            out = subprocess.call(cmd,stderr=subprocess.STDOUT,shell=False)
            if verbose: print('Combined plots saved to '+combname)
        except subprocess.CalledProcessError:
            raise Exception("Could not combine PDFs with ghostscript")
        # Delete individual PDFs
        for fp in pdfnames: os.remove(fp)
    
    # How long did this take
    if verbose is True: print('dt = %5.2f sec.' % (time.time()-t0))
    
    return sumstr, final, bmodel, specmlist



######################### MAIN PROGRAMS ###########################



def fit(spectrum,models=None,fitparams=None,fixparams={},estimates=None,payne=False,
        verbose=False,mcmc=False,figfile=None,cornername=None,retpmodels=False,
        nthreads=None,timestamp=False,notweak=False,tpoly=False,tpolyorder=3):
    """
    Fit the spectrum.  Find the best RV and stellar parameters using the Cannon models.

    Parameters
    ----------
    spectrum : Spec1D object
         The observed spectrum to match.
    models : list of Cannon models or Payne model, optional
         A list of Cannon models or a Payne model to use.  The default is to load
         the models in the data/ directory and use those.
    fitparams : list, optional
         List of Payne labels to fit.
    fixparams : dict, optional
         Dictionary of Payne labels to hold fixed.
    estimates : dict, optional
         Initial estimates for the parameters.
    payne : bool, optional
         Fit a Payne model.  By default, a Cannon model is used.
    verbose : bool, optional
         Verbose output of the various steps.  This is False by default.
    mcmc : bool, optional
         Run Markov Chain Monte Carlo (MCMC) to get improved parameter uncertainties.
         This is False by default.
    figfile : string, optional
         The filename for a diagnostic plot showing the observed spectrum, model
         spectrum and the best-fit parameters.
    cornername : string, optional
         The filename for a "corner" plot showing the posterior distributions from
         the MCMC run.
    retpmodels : bool, optional
         Return the prepared models (only if Cannon models used).
    nthreads : int, optional
         The number of threads to use.  By default the number of threads is not limited.
    timestamp : boolean, optional
         Add timestamp in verbose output (if verbose=True). Default is False.
    notweak : boolean, optional
         Don't tweak the observed continuum using the model.  Default is False.
    tpoly : boolean, optional
         Use a low-order polynomial fit for continuum tweaking.  Default is False,
           it uses Gaussian smoothing instead.
    tpolyorder : int, optional
         Polynomial order to use for continuum tweaking if tpoly=True.
           Default is 3 (cubic).

    Returns
    -------
    out : numpy structured array
         The output structured array of the final derived RVs, stellar parameters and errors.
    model : Spec1D object
         The best-fitting Cannon model spectrum (as Spec1D object).
    specm : Spec1D object
         The observed spectrum with discrepant and outlier pixels masked.
    pmodels : DopplerCannonModelSet
         The prepared Doppler Cannon models, only if retpmodels=True.

    Example
    -------

    .. code-block:: python

         out, model = fit(spec)

    """

    # Set up the logger
    if timestamp and verbose:
        logger = dln.basiclogger()
        logger.handlers[0].setFormatter(logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s"))
        logger.handlers[0].setStream(sys.stdout)
        builtins.logger = logger   # make it available globally across all modules
        
    # Cannon model
    if payne == False:
        out = fit_cannon(spectrum,models=models,estimates=estimates,verbose=verbose,mcmc=mcmc,
                         figfile=figfile,cornername=cornername,retpmodels=retpmodels,
                         nthreads=nthreads,notweak=notweak,tpoly=tpoly,tpolyorder=tpolyorder)
    # Payne model
    else:
        out = fit_payne(spectrum,model=models,fitparams=fitparams,fixparams=fixparams,
                        estimates=estimates,verbose=verbose,mcmc=mcmc,figfile=figfile,
                        cornername=cornername,nthreads=nthreads,
                        notweak=notweak,tpoly=tpoly,tpolyorder=tpolyorder)
        
    # Breakdown logger
    if timestamp and verbose:
        del builtins.logger

    return out
        

def jointfit(speclist,models=None,fitparams=None,fixparams={},mcmc=False,snrcut=10.0,
             saveplot=False,verbose=False,outdir=None,nthreads=None,payne=False,
             timestamp=False,notweak=False,tpoly=False,tpolyorder=3):
    """
    This fits a Cannon or Payne model to multiple spectra of the same star.

    Parameters
    ----------
    speclist : Spec1D object
         The observed spectrum to match.
    models : Cannon or Payne model(s), optional
         A list of Cannon models or Payne model to use.  The default is to load all
         the Cannon or Payne models (depending on if payne=True is set) in the data/
         directory and use those.
    fitparams : list, optional
         List of Payne labels to fit.
    fixparams : dict, optional
         Dictionary of Payne labels to hold fixed.
    mcmc : bool, optional
         Run Markov Chain Monte Carlo (MCMC) to get improved parameter uncertainties.
         This is only run the individual spectra are being fit.
         This is False by default.
    snrcut : int, optional
         S/N cut to fit individual spectra in the first step.  The default is snrcut=10.
    saveplot : bool, optional
         Save output plots.
    verbose : bool, optional
         Verbose output of the various steps.  This is False by default.
    outdir : str, optional
         The directory for output files.  The default is to use the current directory.
    nthreads : int, optional
         The number of threads to use.  By default the number of threads is not limited.
    payne : bool, optional
         Fit a Payne model.  By default, a Cannon model is used.
    timestamp : boolean, optional
         Add timestamp in verbose output (if verbose=True). Default is False.
    notweak : boolean, optional
         Don't tweak the observed continuum using the model.  Default is False.
    tpoly : boolean, optional
         Use a low-order polynomial fit for continuum tweaking.  Default is False,
           it uses Gaussian smoothing instead.
    tpolyorder : int, optional
         Polynomial order to use for continuum tweaking if tpoly=True.
           Default is 3 (cubic).

    Returns
    -------
    sumstr : numpy structured array
         Summary catalog of final best-fit values.
    final : 
         Final best-fit values for each individual spectrum.
    bmodel : List of Spec1D object
         List of best-fitting model spectra.
    specmlist : list of Spec1D object
         List of the observed spectrua with discrepant and outlier pixels masked.

    Example
    -------

    .. code-block:: python

         sumstr, final, bmodel, specmlist = jointfit(speclist)

    """


    # Set up the logger
    if timestamp and verbose:
        logger = dln.basiclogger()
        logger.handlers[0].setFormatter(logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s"))
        logger.handlers[0].setStream(sys.stdout)
        builtins.logger = logger   # make it available globally across all modules
    
    
    # Cannon model
    if payne == False:
        out = jointfit_cannon(speclist,models=models,mcmc=mcmc,snrcut=snrcut,
                              saveplot=saveplot,verbose=verbose,outdir=outdir,
                              nthreads=nthreads,notweak=notweak,tpoly=tpoly,
                              tpolyorder=tpolyorder)
    # Payne model
    else:
        out = jointfit_payne(speclist,model=models,fitparams=fitparams,fixparams=fixparams,
                             mcmc=mcmc,snrcut=snrcut,saveplot=saveplot,verbose=verbose,
                             outdir=outdir,nthreads=nthreads,notweak=notweak,tpoly=tpoly,
                             tpolyorder=tpolyorder)

    # Breakdown logger
    if timestamp and verbose:
        del builtins.logger

    return out
