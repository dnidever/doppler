#!/usr/bin/env python

"""UTILS.PY - Utility functions

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
from scipy.ndimage.filters import median_filter,gaussian_filter1d
from scipy.optimize import curve_fit, least_squares
from scipy.special import erf
from scipy.interpolate import interp1d
#from numpy.polynomial import polynomial as poly
#from lmfit import Model
#from apogee.utils import yanny, apload
#from sdss_access.path import path
import bindata

# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

def lt(x,limit):
    """Takes the lesser of x or limit"""
    if np.array(x).size>1:
        out = [i if (i<limit) else limit for i in x]
    else:
        out = x if (x<limit) else limit
    if type(x) is np.ndarray: return np.array(out)
    return out
    
def gt(x,limit):
    """Takes the greater of x or limit"""
    if np.array(x).size>1:
        out = [i if (i>limit) else limit for i in x]
    else:
        out = x if (x>limit) else limit
    if type(x) is np.ndarray: return np.array(out)
    return out        

def minmax(arr):
    """Return the minimum and maximum"""
    return np.array([np.min(arr),np.max(arr)])

def limit(x,llimit,ulimit):
    """Require x to be within upper and lower limits"""
    return lt(gt(x,llimit),ulimit)
    
def gaussian(x, amp, cen, sig, const=0):
    """1-D gaussian: gaussian(x, amp, cen, sig)"""
    return (amp / (np.sqrt(2*np.pi) * sig)) * np.exp(-(x-cen)**2 / (2*sig**2)) + const

def gaussbin(x, amp, cen, sig, const=0, dx=1.0):
    """1-D gaussian with pixel binning
    
    This function returns a binned Gaussian
    par = [height, center, sigma]
    
    Parameters
    ----------
    x : array
       The array of X-values.
    amp : float
       The Gaussian height/amplitude.
    cen : float
       The central position of the Gaussian.
    sig : float
       The Gaussian sigma.
    const : float, optional, default=0.0
       A constant offset.
    dx : float, optional, default=1.0
      The width of each "pixel" (scalar).
    
    Returns
    -------
    geval : array
          The binned Gaussian in the pixel

    """

    xcen = np.array(x)-cen             # relative to the center
    x1cen = xcen - 0.5*dx  # left side of bin
    x2cen = xcen + 0.5*dx  # right side of bin

    t1cen = x1cen/(np.sqrt(2.0)*sig)  # scale to a unitless Gaussian
    t2cen = x2cen/(np.sqrt(2.0)*sig)

    # For each value we need to calculate two integrals
    #  one on the left side and one on the right side

    # Evaluate each point
    #   ERF = 2/sqrt(pi) * Integral(t=0-z) exp(-t^2) dt
    #   negative for negative z
    geval_lower = erf(t1cen)
    geval_upper = erf(t2cen)

    geval = amp*np.sqrt(2.0)*sig * np.sqrt(np.pi)/2.0 * ( geval_upper - geval_lower )
    geval += const   # add constant offset

    return geval

def gaussfit(x,y,initpar,sigma=None, bounds=None, binned=False):
    """Fit 1-D Gaussian to X/Y data"""
    #gmodel = Model(gaussian)
    #result = gmodel.fit(y, x=x, amp=initpar[0], cen=initpar[1], sig=initpar[2], const=initpar[3])
    #return result
    func = gaussian
    if binned is True: func=gaussbin
    return curve_fit(func, x, y, p0=initpar, sigma=sigma, bounds=bounds)

def poly(x,coef):
    y = np.array(x).copy()*0.0
    for i in range(len(np.array(coef))):
        y += np.array(coef)[i]*np.array(x)**i
    return y

def poly_resid(coef,x,y,sigma=1.0):
    sig = sigma
    if sigma is None: sig=1.0
    return (poly(x,coef)-y)/sig

def poly_fit(x,y,nord,robust=False,sigma=None,bounds=(-np.inf,np.inf)):
    initpar = np.zeros(nord+1)
    # Normal polynomial fitting
    if robust is False:
        #coef = curve_fit(poly, x, y, p0=initpar, sigma=sigma, bounds=bounds)
        weights = None
        if sigma is not None: weights=1/sigma
        coef = np.polyfit(x,y,nord,w=weights)
    # Fit with robust polynomial
    else:
        res_robust = least_squares(poly_resid, initpar, loss='soft_l1', f_scale=0.1, args=(x,y,sigma))
        if res_robust.success is False:
            raise Exception("Problem with least squares polynomial fitting. Status="+str(res_robust.status))
        coef = res_robust.x
    return coef
    
# Derivate of slope of an array
def slope(array):
    """Derivate of slope of an array: slp = slope(array)"""
    n = len(array)
    return array[1:n]-array[0:n-1]

# Median Absolute Deviation
def mad(data, axis=None, func=None, ignore_nan=False):
    """ Calculate the median absolute deviation."""
    if type(data) is not np.ndarray: raise ValueError("data must be a numpy array")    
    return 1.4826 * astropy.stats.median_absolute_deviation(data,axis=axis,func=func,ignore_nan=ignore_nan)

# Gaussian filter
def gsmooth(data,fwhm,axis=-1,mode='reflect',cval=0.0,truncate=4.0):
    return gaussian_filter1d(data,fwhm/2.35,axis=axis,mode=mode,cval=cval,truncate=truncate)

# NUMPY HAS PERCENTILE AND NANPERCENTILE FUNCTIONS!!
# Calculate a percentile value from a given 1D array
#def percentile(arr,percfrac=0.50):
#    '''
#    This calculates a percentile from a 1D array.
#
#    Parameters
#    ----------
#    array : float or int array
#          The array of values.
#    percfrac: float, 0-1
#          The percentage fraction from 0 to 1.
#
#    Returns
#    -------
#    value : float or int
#          The calculated percentile value.
#
#    '''
#    if len(arr)==0: return np.nan
#    si = np.sort(arr)
#    ind = np.int(np.round(percfrac*len(arr)))
#    if ind>(len(arr)-1):ind=len(arr)-1
#    return arr[ind]

# astropy.modeling can handle errors and constraints

def rebin(arr, new_shape):
    if arr.ndim>2:
        raise Exception("Maximum 2D arrays")
    if arr.ndim==0:
        raise Exception("Must be an array")
    if arr.ndim==2:
        shape = (new_shape[0], arr.shape[0] // new_shape[0],
                 new_shape[1], arr.shape[1] // new_shape[1])
        return arr.reshape(shape).mean(-1).mean(1)
    if arr.ndim==1:
        shape = (np.array(new_shape,ndmin=1)[0], arr.shape[0] // np.array(new_shape,ndmin=1)[0])
        return arr.reshape(shape).mean(-1)
