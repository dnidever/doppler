#!/usr/bin/env python

"""LSF.PY - LSF module, classes and functions

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@noao.edu>'
__version__ = '20200112'  # yyyymmdd                                                                                                                           

import numpy as np
import math
import warnings
from scipy.interpolate import interp1d
from scipy import special
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import thecannon as tc
from dlnpyutils import utils as dln, bindata
import copy
from . import utils

_SQRTTWO = np.sqrt(2.)

# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

cspeed = 2.99792458e5  # speed of light in km/s

# astropy.modeling can handle errors and constraints

# The ghlsf(), gausshermitebin(), ghwingsbin() and unpack_ghlsf_params() functions were
# copied from Jo Bovy's "apogee" package which were based on Nidever's code in apogeereduce.

def ghlsf(x,xcenter,params,nowings=False):
    """
    Evaluate the APOGEE Gauss-Hermite LSF (on the native pixel scale).

    Parameters
    ----------
    x : array
      Array of X values for which to compute the LSF (in pixel offset relative to xcenter;
      the LSF is calculated at the x offsets for each xcenter if
      x is 1D, otherwise x has to be [nxcenter,nx]))
    xcenter : array
       Position of the LSF center (in pixel units)
    params : array
       The parameter array (from the LSF HDUs in the APOGEE data products).

    Returns
    -------
    lsf : array
       The 2D LSF array for the input x and xcenter arrays.

    Examples
    --------

    lsf = ghlsf(x,xcenter,params)

    2015-02-26 - Written based on Nidever's code in apogeereduce - Bovy (IAS)
    Heavily modified by D.Nidever to work with Nidever's translated IDL routines Jan 2020.
    """
    # Parse x
    if len(x.shape) == 1:
        x = np.tile(x,(len(xcenter),1))
    # Unpack the LSF parameters
    if type(params) is not dict:
        params = unpack_ghlsf_params(params)
    # Get the wing parameters at each x
    wingparams = np.empty((params['nWpar'],len(xcenter)))
    for ii in range(params['nWpar']):
        poly = np.polynomial.Polynomial(params['Wcoefs'][ii])       
        wingparams[ii] = poly(xcenter+params['Xoffset'])
    # Get the GH parameters at each x
    ghparams = np.empty((params['Horder']+2,len(xcenter)))

    # note that this is modified/corrected a bit from Bovy's routines based on comparison
    # with LSF from IDL routines, noticeable when wings are non-negligible
    for ii in range(params['Horder']+2):
        if ii == 1:
            ghparams[ii] = 1.
        else:
            poly = np.polynomial.Polynomial(params['GHcoefs'][ii-(ii > 1)])
            ghparams[ii] = poly(xcenter+params['Xoffset'])
        # normalization
        if ii > 0: 
            ghparams[ii] /= np.sqrt(2.*np.pi*math.factorial(ii-1))
            if not nowings: ghparams[ii] *= (1.-wingparams[0])

    # Calculate the GH part of the LSF
    # x is [Npix,Nlsf]
    npix = len(xcenter)
    nlsf = x.shape[1]
    xcenter2 = np.tile(xcenter,nlsf).reshape(nlsf,npix).T
    xlsf = x + xcenter2   # absolute X-values
    # Flatten them for gausshermitebin()
    xlsf = xlsf.flatten()
    xcenter2 = xcenter2.flatten()
    # Add in height and center for gausshermitebin() and flatten to 2D
    ghparams1 = np.empty((params['Horder']+4,len(xcenter)))
    ghparams1[0,:] = 1.0
    ghparams1[1,:] = xcenter
    ghparams1[2:,:] = ghparams
    ghparams2 = np.empty((nlsf*npix,params['Horder']+4))
    for i in range(params['Horder']+4):
        ghparams2[:,i] = np.repeat(ghparams1[i,:],nlsf)
    out = gausshermitebin(xlsf,ghparams2,params['binsize'])
    
    # Calculate the Wing part of the LSF
    # Add in center for ghwingsbin() and flatten to 2D
    wingparams1 = np.empty((params['nWpar']+1,len(xcenter)))
    wingparams1[0,:] = wingparams[0,:]
    wingparams1[1,:] = xcenter
    wingparams1[2:,:] = wingparams[1:,:]
    wingparams2 = np.empty((nlsf*npix,params['nWpar']+1))
    for i in range(params['nWpar']+1):
        wingparams2[:,i] = np.repeat(wingparams1[i,:],nlsf)
    if not nowings : out += ghwingsbin(xlsf,wingparams2,params['binsize'],params['Wproftype'])

    # Reshape it to [Npix,Nlsf]
    out = out.reshape(npix,nlsf)
    # Make sure it's positive and normalized
    #out[out<0.]= 0.
    #out /= np.tile(np.sum(out,axis=1),(nlsf,1)).T

    # Flatten for single pixel
    if npix==1: out = out.flatten()
    
    return out


def gausshermitebin(x,par,binsize=1.0):
    """This function returns the value for a superposition of "binned"
    Gauss-Hermite polynomials, or more correctly "Hermite functions".
    These are orthogonal functions.  The "unbinned" version of this
    program is called gausshermite.pro.
    
    The normalization is such that the integral/sum over all pixels is
    equal to par[0]*par[3]*binsize (par[0]*par[3] if /average is set).

    To just scale a profile that has an integral of unity, keep par[3]=1
    and use /average. 

    Currently this only goes up to H4.

    Parameters
    ----------
    x : array
       The array of X-values for which to compute the function values.
       This needs to be 1-dimensional.
    par : array
       The Gauss-Hermite coefficients: [height, center, sigma,
       H0, H1, H2, H3, H4, ...] up to H9.  This needs to be
       2-dimensional [Npar,Nx].
    binsize : float, optional
       The size of each "pixel" in x units.  The default is 1.0.

    Returns
    -------
    y : array
       The output array of the function values for x.

    Examples
    --------

    y = gausshermitebin(x,par)

    By D. Nidever  March/April 2010
    much of it copied from GHM_int.pro by Ana Perez
    """
    
    # This function gives the "binned" flux in a pixel.

    # Parameters are:
    #  3 Gaussian parameters: height, center, sigma
    #  5 Hermite polynomial coefficients

    
    nx = x.shape[0]
    npar = par.shape[1]

    #import pdb; pdb.set_trace()

    
    # HH hermite polynomials
    # GHfunc  the output value
    nherm = npar-3   # 5 is the "standard" way to run it
    hpar = np.zeros((nx,10),np.float64)
    if npar>3:
        hpar[:,0:nherm] = par[:,3:]  # hermite coefficients

    # Initializing arrays
    integ = np.zeros((nx,nherm),np.float64)

    sqrpi = np.sqrt(np.pi)
    sqr2pi = np.sqrt(2*np.pi)

    # Rescale the X-values using Gaussian center and sigma
    #  have two X-values: (1) left side of pixe, (2) right side of pixel
    w = ( x - par[:,1])/par[:,2]
    w1 = ( x-0.5*binsize - par[:,1])/par[:,2]  # previously called "w"
    w2 = ( x+0.5*binsize - par[:,1])/par[:,2]  # previously called "wup"

    # Hermite polynomials are (probabilist's):
    # H0 = 1
    # H1 = x
    # H2 = x^2-1
    # H3 = x^3-3x
    # H4 = x^4-6x^2+3
    # H5 = x^5-10x^3+15x
    # H6 = x^6-15x^4+45x^2-15
    # H7 = x^7-21x^5+105x^3-105x
    # H8 = x^8-28x^6+210x^4-420x^2+105
    # H9 = x^9-36x^7+378x^5-1260x^3+945x
    #
    # So in terms of the powers of X with coefficients
    # Ci for each Hermite polynomial:
    # 0th: C0-C2+3*C4
    # 1st: C1-3*C3
    # 2nd: C2-6*C4
    # 3rd: C3
    # 4th: C4
    # 5th: C5-21C7+378C9
    # 6th: C6-28C8
    # 7th: C7-36C9
    # 8th: C8

    # The Hermite function is:
    #  Psi_n(x) = 1/sqrt(n!*2*pi) * exp(-x^2/2) * H_n(x)

    # So we are doing:
    #  = Integral( P[0]*exp(-0.5*w^2) * SUM_i( P[i+3]*H[i,w] / sqrt(i!*2*pi) ) )
    #  where H are the hermite polynomials, and the Sum over i is from 0->4
    #  the integral is from x1->x2, w=(x-P[1])/P[2]

    # The normalization is sqrt(n!)
    # 0th: 1
    # 1st: 1
    # 2nd: sqrt(2)
    # 3rd: sqrt(6)
    # 4th: sqrt(24)
    # 5th: sqrt(120)
    # 6th: sqrt(720)
    # 7th: sqrt(5040)
    # 8th: sqrt(40320)
    # 9th: sqrt(362880)
    sqr2 = np.sqrt(2)  # some sqrt constants
    sqr3 = np.sqrt(3)
    sqr6 = np.sqrt(6)
    sqr24 = np.sqrt(24)
    sqr120 = np.sqrt(120)
    sqr720 = np.sqrt(720)
    sqr5040 = np.sqrt(5040)
    sqr40320 = np.sqrt(40320)
    sqr362880 = np.sqrt(362880)

    # HH are the coefficients for the powers of X
    #   hpar will be zero for orders higher than are actually desired
    hh = np.zeros((nx,10),np.float64)      # the coefficients for the powers of X
    hh[:,0] = hpar[:,0] - hpar[:,2]/sqr2 + hpar[:,4]*3/sqr24 - hpar[:,6]*15/sqr720 + hpar[:,8]*105/sqr40320
    hh[:,1] = hpar[:,1] - hpar[:,3]*3/np.sqrt(6) + hpar[:,5]*15/sqr120 - hpar[:,7]*105/sqr5040 + hpar[:,9]*945/sqr362880
    hh[:,2] = hpar[:,2]/sqr2 - hpar[:,4]*(6/sqr24) + hpar[:,6]*45/sqr720 - hpar[:,8]*420/sqr40320
    hh[:,3] = hpar[:,3]/np.sqrt(6) - hpar[:,5]*10/sqr120 + hpar[:,7]*105/sqr5040 - hpar[:,9]*1260/sqr362880
    hh[:,4] = hpar[:,4]/sqr24 - hpar[:,6]*15/sqr720 + hpar[:,8]*210/sqr40320
    hh[:,5] = hpar[:,5]/sqr120 - hpar[:,7]*21/sqr5040 + hpar[:,9]*378/sqr362880
    hh[:,6] = hpar[:,6]/sqr720 - hpar[:,8]*28/sqr40320
    hh[:,7] = hpar[:,7]/sqr5040 - hpar[:,9]*36/sqr362880
    hh[:,8] = hpar[:,8]/sqr40320
    hh[:,9] = hpar[:,9]/sqr362880

    # Gaussian values at the edges
    eexp1 = np.exp(-0.5*w1**2)
    eexp2 = np.exp(-0.5*w2**2)
    
    # Integrals of exp(-w^2/2)*w^i:
    #  i=0   sqrt(pi/2) * erf(w^2/sqrt(2)) 
    #  i=1   -exp(-w^2/w)
    #  i=2   -w*exp(-w^2/2) * Integral( exp(-w^2/2) )
    #  i=3   -w^2*exp(-w^2/2) * 2*Integral( w*exp(-w^2/2) )
    #  i=4   -w^3*exp(-w^2/2) * 3*Integral( w^2*exp(-w^2/2) )
    #  i=5   and so on
    #  i=6   
    #  i=7   
    #  i=8   
    #  after i=1 we can use a recursion relation
    integ[:,0] = np.sqrt(np.pi/2)*(special.erf(w2/sqr2) - special.erf(w1/sqr2))
    if nherm>1: integ[:,1] = -eexp2 + eexp1
    if nherm>2:
        for i in range(2,nherm-1):
            integ[:,i] = ( -w2**(i-1)*eexp2 + w1**(i-1)*eexp1 ) + (i-1)*integ[:,i-2]

    #  Now multiply times the polynomial coefficients and sum
    GHfunc = np.zeros(nx,np.float64)
    for i in range(0,nherm-1):
        GHfunc += hh[:,i]*integ[:,i]
    GHfunc *= par[:,0]/sqr2pi


    return GHfunc


def ghwingsbin(x,par,binsize=1.0,Wproftype=1):
    """This function returns profile wings with the APOGEE LSF or PSF.

    The normalization is such that the integral/sum over all pixels is
    equal to par[0]*par[3].

    To just scale a profile that has an integral of unity, keep par[3]=1.

    Parameters
    ----------
    x : array
      The array of X-values for which to compute the function values.
      This needs to be 1-dimensional.
    par  : array
       The parameters.  The 1st gives the "area under the curve", the
       normalization.  The 2nd gives the center of the
       profile.  The meaning of the other paramters depend on
       the profile type.
       This needs to be 2-dimensional [Nx,Npar].
    binsize : float, optional
        The size of each "pixel" in X units.  The default is 1.0.
    proftype : int
          The profile type:
                 1-Gaussian
                 2-Lorentzian
                 3-Exponential
                 4-1/r^2
                 5-1/r^3

    Returns
    -------
    y : array
      The output array of the function values for x.

    Examples
    --------

    y = ghwingsbin(x,par,binsize=1,1)

    By D. Nidever  April 2011
    translated to python Jan 2020
    """

    nx = x.shape[0]
    npar = par.shape[1]

    # The profile types:
    #   1-Gaussian
    #   2-Lorentzian
    #   3-Exponential
    #   4-1/r^2

    # Parameters
    # The 1st parameter is always the "area under the curve"
    # and gives the total normalization.
    # The 2nd paramter is always the CENTER.
    # The meaning of the other parameters depend on
    # profile type.

    # 1. Gaussian
    #--------------
    if (Wproftype==1):
        if npar<3:
            raise ValueError('Need 3 parameters for Gaussian')
        
        # y = A*exp(-0.5*(x-center)^2/sigma^2)
        # Parameters are: [normalization, center, sigma]
        # Gaussian area = ht*wid*sqrt(2*pi)
        center = par[:,1]
        sigma = par[:,2]
        # Rescale the X-values using Gaussian center and sigma
        #  have two X-values: (1) left side of pixel, (2) right side of pixel
        w = ( x - par[:,1])/par[:,2]
        w1 = ( x-0.5*binsize - par[:,1])/par[:,2]
        w2 = ( x+0.5*binsize - par[:,1])/par[:,2]
        sqr2 = np.sqrt(2)  # some sqrt constants

        y = par[:,0]*0.5*(special.erf(w2/sqr2) - special.erf(w1/sqr2))
        # erf returns a normalized gaussian (2x) so we didn't
        # need to scale the amplitude
        return y

    # 2. Lorentzian
    #--------------
    if (Wproftype==2):
        if npar<3:
            raise ValueError('Need 3 parameters for Lorentzian')

        # y = A / ( ((x-center)/sigma)^2 + 1)

        # Parameters are: [normalization, center, sigma]
        # Area = ht*sigma*pi
        center = par[:,1]
        sigma = par[:,2]

        # Rescale the X-values using Gaussian center and sigma
        #  have two X-values: (1) left side of pixel, (2) right side of pixel
        w = ( x - par[:,1])/par[:,2]
        w1 = ( x-0.5*binsize - par[:,1])/par[:,2]
        w2 = ( x+0.5*binsize - par[:,1])/par[:,2]
        sqr2 = np.sqrt(2)  # some sqrt constants

        y = par[:,0]*(np.atan(w2) - np.atan(w1))/np.pi
        return y

    # 3. Exponential
    #---------------
    if (Wproftype==3):
        if npar<3:
            raise ValueError('Need 3 parameters for Exponential')

        # y = A*exp(-abs(x-center)/scale)
        #
        # Parameters are: [normalization, center, scale]
        # Area = 2*A*scale
        center = par[:,1]
        scale = par[:,2]
        #ht = 2*par[0]*scale
        #y = ht * exp(-abs(x-center)/scale)

        # Rescale the X-values using Gaussian center and sigma
        #  have two X-values: (1) left side of pixel, (2) right side of pixel
        w = ( x - par[:,1])/par[:,2]
        w1 = ( x-0.5*binsize - par[:,1])/par[:,2]
        w2 = ( x+0.5*binsize - par[:,1])/par[:,2]
        sqr2 = np.sqrt(2)  # some sqrt constants

        y = 2*par[:,0]*scale**2*(-np.exp(-np.abs(w2))*dln.signs(w2) + np.exp(-np.abs(w1))*dln.signs(w1))

        # this has a problem at x=xcenter
        # Need to treat any points across x=xcenter separately
        cross, = np.where( ((w1 < 0) & (w2 > 0)) | (w1 == 0) | (w2 == 0))
        ncross = dln.size(cross)
        if ncross>0:
            y[cross] = 2*par[:,0]*scale**2*(1 - np.exp(-np.abs(w2[cross]))) + \
                       2*par[:,0]*scale**2*(1 - np.exp(-np.abs(w1[cross])))

    # 4. 1/r^2
    #----------
    if (Wproftype==4):
        if npar<2:
            raise ValueError('Need 2 parameters for 1/r^2')

        # y = A / ( (x-center)^2 + 1)
        # very similar to lorentzian, with sigma=1

        # Parameters are: [nxormalization, center]
        # Area = ht*pi
        center = par[:,1]
        #ht = par[0]/!dpi
        #y = ht / ( (x-center)^2 + 1d )

        # Rescale the X-values using Gaussian center and sigma
        #  have two X-values: (1) left side of pixel, (2) right side of pixel
        w = ( x - par[:,1])
        w1 = ( x-0.5*binsize - par[:,1])
        w2 = ( x+0.5*binsize - par[:,1])
        sqr2 = np.sqrt(2)  # some sqrt constants
        
        y = par[:,0]*(np.atan(w2) - np.atan(w1))/np.pi
        return y



# The _bovy functions below are much slower because they haven't
# been completely vectorized.

def ghlsf_bovy(x,xcenter,params,nowings=False):
    """
    Evaluate the raw APOGEE Gauss-Hermite LSF (on the native pixel scale)
    
    Parameters
    ----------
    x : array
       Array of X values for which to compute the LSF (in pixel offset relative to xcenter;
       the LSF is calculated at the x offsets for each xcenter if
       x is 1D, otherwise x has to be [nxcenter,nx]))
    xcenter : array
        Position of the LSF center (in pixel units).
    params : array
        The parameter array (from the LSF HDUs in the APOGEE data products).
    nowings : bool, optional
       Do not include the wings.  The default is False.

    Returns
    -------
    lsf : array
       The LSF array for each xcenter.

    Examples
    --------

    lsf = ghlsf_bovy(x,xcenter,params)

    2015-02-26 - Written based on Nidever's code in apogeereduce - Bovy (IAS)
    """

    # Parse x
    if len(x.shape) == 1:
        x = np.tile(x,(len(xcenter),1))
    # Unpack the LSF parameters
    if type(params) is not dict:
        params = unpack_ghlsf_params(params)
    # Get the wing parameters at each x
    wingparams = np.empty((params['nWpar'],len(xcenter)))
    for ii in range(params['nWpar']):
        poly = np.polynomial.Polynomial(params['Wcoefs'][ii])       
        wingparams[ii] = poly(xcenter+params['Xoffset'])
    # Get the GH parameters at each x
    ghparams = np.empty((params['Horder']+2,len(xcenter)))

    # note that this is modified/corrected a bit from Bovy's routines based on comparison
    # with LSF from IDL routines, noticeable when wings are non-negligible
    for ii in range(params['Horder']+2):
        if ii == 1:
            ghparams[ii] = 1.
        else:
            poly = np.polynomial.Polynomial(params['GHcoefs'][ii-(ii > 1)])
            ghparams[ii] = poly(xcenter+params['Xoffset'])
        # normalization
        if ii > 0: 
            ghparams[ii] /= np.sqrt(2.*np.pi*math.factorial(ii-1))
            if not nowings: ghparams[ii] *= (1.-wingparams[0])
    # Calculate the GH part of the LSF
    out = gausshermitebin_bovy(x,ghparams,params['binsize'])
    # Calculate the Wing part of the LSF
    if not nowings: out += ghwingsbin_bovy(x,wingparams,params['binsize'],params['Wproftype'])
    
    return out


def gausshermitebin_bovy(x,params,binsize=1.0):
    """Evaluate the integrated APOGEE Gauss-Hermite function"""
    ncenter = params.shape[1]
    out = np.empty((ncenter,x.shape[1]))
    integ = np.empty((params.shape[0]-1,x.shape[1]))
    for ii in range(ncenter):
        poly = np.polynomial.HermiteE(params[1:,ii])
        # Convert to regular polynomial basis for easy integration
        poly = poly.convert(kind=np.polynomial.Polynomial)
        # Integrate and add up
        w1 = (x[ii]-0.5*binsize)/params[0,ii]
        w2 = (x[ii]+0.5*binsize)/params[0,ii]
        eexp1 = np.exp(-0.5*w1**2.)
        eexp2 = np.exp(-0.5*w2**2.)
        integ[0] = np.sqrt(np.pi/2.)\
            *(special.erf(w2/_SQRTTWO)-special.erf(w1/_SQRTTWO))
        out[ii] = poly.coef[0]*integ[0]
        if params.shape[0] > 1:
            integ[1] = -eexp2+eexp1
            out[ii] += poly.coef[1]*integ[1]
        for jj in range(2,params.shape[0]-1):
            integ[jj] = (-w2**(jj-1)*eexp2+w1**(jj-1)*eexp1)\
                +(jj-1)*integ[jj-2]
            out[ii] += poly.coef[jj]*integ[jj]
    return out


def ghwingsbin_bovy(x,params,binsize,Wproftype):
    """Evaluate the wings of the APOGEE LSF"""
    ncenter = params.shape[1]
    out = np.empty((ncenter,x.shape[1]))
    for ii in range(ncenter):
        if Wproftype == 1: # Gaussian
            w1 = (x[ii]-0.5*binsize)/params[1,ii]
            w2 = (x[ii]+0.5*binsize)/params[1,ii]
            out[ii] = params[0,ii]/2.*(special.erf(w2/_SQRTTWO) \
                                       -special.erf(w1/_SQRTTWO))
    return out


def unpack_ghlsf_params(lsfarr):
    """
    Unpack the APOGEE Gauss-Hermite LSF parameter array into its constituents.

    Parameters
    ----------
    lsfarr : array


    Returns
    -------
    params : dictionary
        Dictionary with unpacked parameters and parameter values: 
          binsize: The width of a pixel in X-units
          Xoffset: An additive x-offset; used for GH parameters that vary globally
          Horder: The highest Hermite order
          Porder: Polynomial order array for global variation of each LSF parameter
          GHcoefs: Polynomial coefficients for sigma and the Horder Hermite parameters
          Wproftype: Wing profile type
          nWpar: Number of wing parameters
          WPorder: Polynomial order for the global variation of each wing parameter          
          Wcoefs: Polynomial coefficients for the wings parameters

    Examples
    --------

    params = unpack_ghlsf_params(lsfarr)

    2015-02-15 - Written based on Nidever's code in apogeereduce - Bovy (IAS@KITP)
    """
    
    # If 2D then iterate over orders and return list
    if lsfarr.ndim==2:
        nx,ny = lsfarr.shape
        if nx<ny:
            lsfarr = lsfarr.T
            nx,ny = lsfarr.shape
        out = []
        for i in range(ny):
            out.append(unpack_ghlsf_params(lsfarr[:,i]))
        return out

    # Process 1D array
    out = {}
    # binsize: The width of a pixel in X-units
    out['binsize'] = lsfarr[0]
    # X0: An additive x-offset; used for GH parameters that vary globally
    out['Xoffset'] = lsfarr[1]
    # Horder: The highest Hermite order
    out['Horder'] = int(lsfarr[2])
    # Porder: Polynomial order array for global variation of each LSF parameter
    out['Porder'] = lsfarr[3:out['Horder']+4]
    out['Porder'] = out['Porder'].astype('int')
    nGHcoefs = np.sum(out['Porder']+1)
    # GHcoefs: Polynomial coefficients for sigma and the Horder Hermite parameters
    maxPorder = np.amax(out['Porder'])
    GHcoefs = np.zeros((out['Horder']+1,maxPorder+1))
    GHpar = lsfarr[out['Horder']+4:out['Horder']+4+nGHcoefs] #all coeffs
    CoeffStart = np.hstack((0,np.cumsum(out['Porder']+1)))
    for ii in range(out['Horder']+1):
        GHcoefs[ii,:out['Porder'][ii]+1] = GHpar[CoeffStart[ii]:CoeffStart[ii]+out['Porder'][ii]+1]
    out['GHcoefs'] = GHcoefs
    # Wproftype: Wing profile type
    wingarr = lsfarr[3+out['Horder']+1+nGHcoefs:]
    out['Wproftype'] = int(wingarr[0])
    # nWpar: Number of wing parameters
    out['nWpar'] = int(wingarr[1])
    # WPorder: Polynomial order for the global variation of each wing parameter
    out['WPorder'] = wingarr[2:2+out['nWpar']]
    out['WPorder'] = out['WPorder'].astype('int')
    # Wcoefs: Polynomial coefficients for the wings parameters
    maxWPorder = np.amax(out['WPorder'])
    Wcoefs = np.zeros((out['nWpar'],maxWPorder+1))
    Wpar = wingarr[out['nWpar']+2:]
    WingcoeffStart= np.hstack((0,np.cumsum(out['WPorder']+1)))
    for ii in range(out['nWpar']):
        Wcoefs[ii,:out['WPorder'][ii]+1] = Wpar[WingcoeffStart[ii]:WingcoeffStart[ii]+out['WPorder'][ii]+1]
    out['Wcoefs'] = Wcoefs
    return out


# Base class for representing LSF (line spread function)
class Lsf:
    """
    A base class for representing Line Spread Functions (LSF).
    The subclasses GaussianLsf or GaussHermiteLsf should actually be used in practice.

    Either the LSF coefficients must be input (with pars and xtype) or the array of Gaussian
    sigma values (using sigma).

    Parameters
    ----------
    wave : array, optional
        Wavelength array.
    pars : array, optional
        The LSF coefficients giving the LSF as a function of wavelength or pixels (specified in xtype).
    xtype : string, optional
        The dependence of pars.  Either 'wave' or 'pixel'.
    lsftype : string, optional
        Type of LSF: 'Gaussian' or 'Gauss-Hermite'.
    sigma : array, optional
        Array of Gaussian sigmas for all the pixels.
    verbose : bool, optional
      Verbose output.  False by default.

    """
    
    # Initalize the object
    def __init__(self,wave=None,pars=None,xtype='wave',lsftype='Gaussian',sigma=None,verbose=False):
        # xtype is wave or pixels.  designates what units to use BOTH for the input
        #   arrays to use with PARS and the output units
        if wave is None and xtype=='Wave':
            raise Exception('Need wavelength information if xtype=Wave')
        self.wave = wave
        self.pars = pars
        self.lsftype = lsftype
        self.xtype = xtype
        if wave.ndim==1:
            npix = len(wave)
            norder = 1
        else:
            npix,norder = wave.shape
        self.ndim = wave.ndim
        self.npix = npix
        self.norder = norder
        self._sigma = sigma
        self._array = None
        if (pars is None) & (sigma is None):
            if verbose is True: print('No LSF information input.  Assuming Nyquist sampling.')
            # constant FWHM=2.5, sigma=2.5/2.35
            self.pars = np.array([2.5 / 2.35])
            self.xtype = 'Pixels'

            
    def wave2pix(self,w,extrapolate=True,order=0):
        """
        Convert wavelength values to pixels using the LSF's dispersion
        solution.

        Parameters
        ----------
        w : array
          Array of wavelength values to convert.
        extrapolate : bool, optional
           Extrapolate beyond the boundaries of the dispersion solution,
           if necessary.  True by default.
        order : int, optional
            The order to use if there are multiple orders.
            The default is 0.
              
        Returns
        -------
        x : array
          The array of converted pixels.

        Examples
        --------

        x = lsf.wave2pix(w)

        """
               
        if self.wave is None:
            raise Exception("No wavelength information")
        if self.ndim==2:
            # Order is always the second dimension
            return utils.w2p(self.wave[:,order],w,extrapolate=extrapolate)            
        else:
            return utils.w2p(self.wave,w,extrapolate=extrapolate)

        
    def pix2wave(self,x,extrapolate=True,order=0):
        """
        Convert pixel values to wavelengths using the LSF's dispersion
        solution.

        Parameters
        ----------
        x : array
          Array of pixel values to convert.
        extrapolate : bool, optional
           Extrapolate beyond the boundaries of the dispersion solution,
           if necessary.  True by default.
        order : int, optional
            The order to use if there are multiple orders.
            The default is 0.
              
        Returns
        -------
        w : array
          The array of converted wavelengths.

        Examples
        --------

        w = lsf.pix2wave(x)

        """
        
        if self.wave is None:
            raise Exception("No wavelength information")
        if self.ndim==2:
             # Order is always the second dimension
            return utils.p2w(self.wave[:,order],x,extrapolate=extrapolate)
        else:
            return utils.p2w(self.wave,x,extrapolate=extrapolate)        

        
    # Return FWHM at some positions
    def fwhm(self,x=None,xtype='pixels',order=0):
        """
        Compute the Full-Width at Half-Maximum (FWHM) of the LSF at specific
        locations along the spectrum.

        Parameters
        ----------
        x : array, optional
          The array of x-values (either wavelength or pixels) for which to
          compute the FWHM.  The default is compute it for the entire spectrum.
        xtype : string, optional
          The type of x-value in put (either 'wave' or 'pixels').  The default
          is 'pixels'.
        order : int, optional
          The spectral order to use, if there are multiple orders.  The default is 0.

        Returns
        -------
        fwhm : array
           The array of FWHM values.

        Examples
        --------

        fwhm = lsf.fwhm()

        """
        return self.sigma(x,xtype=xtype,order=order)*2.35

    
    # Return Gaussian sigma. Must be defined by subclass.
    def sigma(self):
        """ Returns the Gaussian sigma.  Must be defined by the subclass."""
        pass

    
    # Clean up bad LSF values. Must be defined by subclass.
    def clean(self):
        """ Clean up bad LSF values. Must be defined by subclass."""
        pass

    
    # Return full LSF values for the spectrum. Must be defined by subclass.
    def array(self):
        """ Return full LSF values for the spectrum. Must be defined by subclass."""
        pass

    
    # Return full LSF values using contiguous input array.  Must be defined by subclass.
    def anyarray(self):
        """ Return full LSF values using contiguous input array.  Must be defined by subclass."""
        pass

    
    def copy(self):
        """ Create a new copy of this LSF object."""
        return copy.deepcopy(self)



# Class for representing Gaussian LSFs
class GaussianLsf(Lsf):
    """
    A class for representing Gaussian Line Spread Functions (LSF).

    Either the LSF coefficients must be input (with pars and xtype) or the array of Gaussian
    sigma values (using sigma).

    Parameters
    ----------
    wave : array, optional
        Wavelength array.
    pars : array, optional
        The LSF coefficients giving the LSF as a function of wavelength or pixels (specified in xtype).
    xtype : string, optional
        The dependence of pars.  Either 'wave' or 'pixel'.
    sigma : array, optional
        Array of Gaussian sigmas for all the pixels.
    verbose : bool, optional
      Verbose output.  False by default.

    The input below are not used and only for consistency with the other LSF classes.
    lsftype : LSF type

    """

    # Initalize the object
    def __init__(self,wave=None,pars=None,xtype='wave',lsftype='Gaussian',sigma=None,verbose=False):
        # xtype is wave or pixels.  designates what units to use BOTH for the input
        #   arrays to use with PARS and the output units
        if wave is None and xtype=='Wave':
            raise Exception('Need wavelength information if xtype=Wave')
        super().__init__(wave=wave,pars=pars,xtype=xtype,lsftype='Gaussian',sigma=sigma,verbose=False)        

        
    # Return Gaussian sigma
    def sigma(self,x=None,xtype='pixels',order=0,extrapolate=True):
        """
        Return the Gaussian sigma at specified locations.  The sigma
        will be in units of lsf.xtype.

        Parameters
        ----------
        x : array, optional
          The x-values for which to return the Gaussian sigma values.
        xtype : string, optional
           The type of x-value input, either 'wave' or 'pixels'.  Default is 'pixels'.
        order : int, optional
            The order to use if there are multiple orders.
            The default is 0.
        extrapolate : bool, optional
           Extrapolate beyond the dispersion solution, if necessary.
           True by default.

        Returns
        -------
        sigma : array
            The array of Gaussian sigma values.

        Examples
        --------
        sigma = lsf.sigma([100,200])
        
        """
        
        # The sigma will be returned in units given in lsf.xtype
        if self._sigma is not None:
            _sigma = self._sigma
            if self.ndim==2: _sigma = self._sigma[:,order]
            if x is None:
                return _sigma
            else:
                # Wavelength input
                if xtype.lower().find('wave') > -1:
                    x0 = np.array(x).copy()            # backup
                    x = self.wave2pix(x0,order=order)  # convert to pixels
                # Integer, just return the values
                if( type(x)==int) | (np.array(x).dtype.kind=='i'):
                    return _sigma[x]
                # Floats, interpolate
                else:
                    sig = interp1d(np.arange(len(_sigma)),_sigma,kind='cubic',bounds_error=False,
                                   fill_value=(np.nan,np.nan),assume_sorted=True)(x)
                    # Extrapolate
                    npix = self.npix
                    if ((np.min(x)<0) | (np.max(x)>(npix-1))) & (extrapolate is True):
                        xin = np.arange(npix)
                        # At the beginning
                        if (np.min(x)<0):
                            coef1 = dln.poly_fit(xin[0:10], _sigma[0:10], 2)
                            bd1, nbd1 = dln.where(x <0)
                            sig[bd1] = dln.poly(x[bd1],coef1)
                        # At the end
                        if (np.max(x)>(npix-1)):
                            coef2 = dln.poly_fit(xin[npix-10:], _sigma[npix-10:], 2)
                            bd2, nbd2 = dln.where(x > (npix-1))
                            sig[bd2] = dln.poly(x[bd2],coef2)
                    return sig
                        
        # Need to calculate
        else:
            if x is None:
                x = np.arange(self.npix)
            if self.pars is None:
                raise Exception("No LSF parameters")
            # Get parameters
            pars = self.pars
            if self.ndim==2: pars=self.pars[:,order]
            # Pixels input
            if xtype.lower().find('pix') > -1:
                # Pixel LSF parameters
                if self.xtype.lower().find('pix') > -1:
                    return np.polyval(pars[::-1],x)
                # Wave LSF parameters
                else:
                    w = self.pix2wave(x,order=order)
                    return np.polyval(pars[::-1],w)                    
            # Wavelengths input
            else:
                # Wavelength LSF parameters
                if self.xtype.lower().find('wave') > -1:
                    return np.polyval(pars[::-1],x)
                # Pixel LSF parameters
                else:
                    x0 = np.array(x).copy()
                    x = self.wave2pix(x0,order=order)
                    return np.polyval(pars[::-1],x)  

                
    # Clean up bad LSF values
    def clean(self):
        """ Clean up bad input Gaussian sigma values."""
        if self._sigma is not None:
            smlen = np.round(self.npix // 50).astype(int)
            if smlen==0: smlen=3
            _sigma = self._sigma.reshape(self.npix,self.norder)   # make 2D
            for o in range(self.norder):
                sig = _sigma[:,o]
                smsig = dln.gsmooth(sig,smlen)
                bd,nbd = dln.where(sig <= 0)
                if nbd>0:
                    sig[bd] = smsig[bd]
                if self.ndim==2:
                    self._sigma[:,o] = sig
                else:
                    self._sigma = sig

                    
    # Return full LSF values for the spectrum
    def array(self,order=None):
        """
        Return the full LSF for the spectrum.

        Parameters
        ----------
        order : int, optional
           The order for which to return the full LSF array, if there are multiple
           orders.  The default is None which means the LSF array for all orders is
           returned.

        Returns
        -------
        lsf : array
           The full LSF array for the spectrum (or just one order).

        Examples
        --------
        lsf = lsf.array()

        """
        
        # Return what we already have
        if self._array is not None:
            if (self.ndim==2) & (order is not None):
                # [Npix,Nlsf,Norder]
                return self._array[:,:,order]
            else:
                return self._array

        # Loop over orders and figure out how many Nlsf pixels we need
        #  must be same across all orders
        wave = self.wave.reshape(self.npix,self.norder)
        nlsfarr = np.zeros(self.norder,dtype=int)
        xsigma = np.zeros((self.npix,self.norder),dtype=np.float64)
        for o in range(self.norder):
            x = np.arange(self.npix)
            xsig = self.sigma(order=o)
            w = wave[:,o]
            
            # Convert sigma from wavelength to pixels, if necessary
            if self.xtype.lower().find('wave') > -1:
                wsig = xsig.copy()
                dw = dln.slope(w)
                dw = np.hstack((dw,dw[-1]))            
                xsig = wsig / dw
            xsigma[:,o] = xsig
                
            # Figure out nLSF pixels needed, +/-3 sigma
            nlsf = np.int(np.round(np.max(xsigma)*6))
            if nlsf % 2 == 0: nlsf+=1                   # must be odd
            nlsfarr[o] = nlsf
        nlsf = np.max(np.array(nlsfarr))

        # Make LSF array
        lsf = np.zeros((self.npix,nlsf,self.norder))
        # Loop over orders
        for o in range(self.norder):
            xlsf = np.arange(nlsf)-nlsf//2
            xlsf2 = np.repeat(xlsf,self.npix).reshape((nlsf,self.npix)).T
            lsf1 = np.zeros((self.npix,nlsf))
            xsig = xsigma[:,o]
            xsigma2 = np.repeat(xsig,nlsf).reshape((self.npix,nlsf))
            lsf1 = np.exp(-0.5*xlsf2**2 / xsigma2**2) / (np.sqrt(2*np.pi)*xsigma2)
            # should I use gaussbin????
            lsf1[lsf1<0.] = 0.
            lsf1 /= np.tile(np.sum(lsf1,axis=1),(nlsf,1)).T
            lsf[:,:,o] = lsf1
            
        # if only one order then reshape
        if self.ndim==1: lsf=lsf.reshape(self.npix,nlsf)
            
        self._array = lsf   # save for next time

        return lsf

    
    # Return LSF values using contiguous input array
    def anyarray(self,x,xtype='pixels',order=0,original=True):
        """
        Return the LSF of the spectrum at specific locations or
        on a new wavelength array.

        Parameters
        ----------
        x : array
          The array of x-values for which to return the LSF.
        xtype : string, optional
          The type of x-values input, either 'wave' or 'pixels'.  The default
          is 'pixels'.
        order : int, optional
           The order for which to retrn the LSF array.  The default is 0.
        original : bool, optional
           If original=True, then the LSFs are returned on the original
           wavelength scale but at the centers given in "x".
           If the LSF is desired on a completely new wavelength scale
           (given by "x"), then orignal=False should be used instead.

        Returns
        -------
        lsf : array
           The 2D array of LSF values.

        Examples
        --------
        lsf = lsf.anyarray([100,200])

        """

        nx = len(x)
        # returns xsigma in units of self.xtype not necessarily xtype
        xsigma = self.sigma(x,xtype=xtype,order=order)
        
        # Get wavelength and pixel arrays
        if xtype.lower().find('pix') > -1:
            w = self.pix2wave(x,order=order)
        else:
            w = x.copy()
            x = self.wave2pix(w,order=order)
            
        # Convert sigma from wavelength to pixels, if necessary
        if self.xtype.lower().find('wave') > -1:
            wsigma = xsigma.copy()
            # On original wavelength scale
            if original is True:
                w1 = self.pix2wave(np.array(x)+1,order=order)
                dw = w1-w
            # New wavelength/pixel scale
            else:
                dw = dln.slope(w)
                dw = np.hstack((dw,dw[-1]))            
            xsigma = wsigma / dw

        # Figure out nLSF pixels needed, +/-3 sigma
        nlsf = np.int(np.round(np.max(xsigma)*6))
        if nlsf % 2 == 0: nlsf+=1                   # must be odd
        
        # Make LSF array
        lsf = np.zeros((nx,nlsf))
        xlsf = np.arange(nlsf)-nlsf//2
        xlsf2 = np.repeat(xlsf,nx).reshape((nlsf,nx)).T
        xsigma2 = np.repeat(xsigma,nlsf).reshape((nx,nlsf))
        lsf = np.exp(-0.5*xlsf2**2 / xsigma2**2) / (np.sqrt(2*np.pi)*xsigma2)
        lsf[lsf<0.] = 0.
        lsf /= np.tile(np.sum(lsf,axis=1),(nlsf,1)).T
        
        # should I use gaussbin????
        return lsf


# Class for representing Gauss-Hermite LSFs
class GaussHermiteLsf(Lsf):
    """
    A class for representing Gauss-Hermite Line Spread Functions (LSF).

    Parameters
    ----------
    wave : array
        Wavelength array.
    pars : array
        The LSF coefficients giving the Gauss-Hermite LSF as a function of pixels.
    verbose : bool, optional
      Verbose output.  False by default.

    The inputs below are not used and only for consistency with the other LSF classes.
    lsftype : LSF type
    xtype : type of parameter coefficients.
    sigma : Gaussian sigma array

    """
    
    # Initalize the object
    def __init__(self,wave=None,pars=None,xtype='pixel',lsftype='Gauss-Hermite',sigma=None,verbose=False):
        # xtype is wave or pixels.  designates what units to use BOTH for the input
        #   arrays to use with PARS and the output units
        if wave is None and xtype=='Wave':
            raise Exception('Need wavelength information if xtype=Wave')
        super().__init__(wave=wave,pars=pars,xtype='pixel',lsftype='Gauss-Hermite',sigma=sigma,verbose=False)

        
    # Return Gaussian sigma
    def sigma(self,x=None,xtype='pixels',order=0,extrapolate=True):
        """
        Return the Gaussian sigma at specified locations.  The sigma
        will be in pixel units.

        Parameters
        ----------
        x : array, optional
          The x-values for which to return the Gaussian sigma values.
        xtype : string, optional
           The type of x-value input, either 'wave' or 'pixels'.  Default is 'pixels'.
        order : int, optional
            The order to use if there are multiple orders.
            The default is 0.
        extrapolate : bool, optional
           Extrapolate beyond the dispersion solution, if necessary.
           True by default.

        Returns
        -------
        sigma : array
            The array of Gaussian sigma values in units of xtype.

        Examples
        --------
        sigma = lsf.sigma([100,200])
        
        """
        
        # The sigma will be returned in pixels
        if x is None:
            x = np.arange(self.npix)
        if self.pars is None:
            raise Exception("No LSF parameters")
        # Get parameters
        pars = self.pars
        if self.ndim==2: pars=self.pars[:,order]
        # Wavelengths input
        if xtype.lower().find('wave') > -1:
            w = x.copy()
            x = self.wave2pix(w,order=order)
        # Compute sigma
        params = unpack_ghlsf_params(pars)
        # Get the GH parameters at each x for sigma
        poly = np.polynomial.Polynomial(params['GHcoefs'][0])
        sigma = poly(x+params['Xoffset'])
        return sigma

    
    # Clean up bad LSF values
    def clean(self):
        """ Clean the LSF values.  This is not implemented for Gauss-Hermite LSF."""
        # No cleaning for now
        pass

    
    # Return full LSF values for the spectrum
    def array(self,order=None):
        """
        Return the full LSF for the spectrum.

        Parameters
        ----------
        order : int, optional
           The order for which to return the full LSF array, if there are multiple
           orders.  The default is None which means the LSF array for all orders is
           returned.

        Returns
        -------
        lsf : array
           The full LSF array for the spectrum (or just one order).

        Examples
        --------
        lsf = lsf.array()

        """
        
        xtype = 'pixels'     # only pixels supported for now
        # Return what we already have
        if self._array is not None:
            if (self.ndim==2) & (order is not None):
                # [Npix,Nlsf,Norder]
                return self._array[:,:,order]
            else:
                return self._array

        # Make LSF array
        nlsf = 15    # default, for now
        lsf = np.zeros((self.npix,nlsf,self.norder))
        xlsf = np.arange(nlsf)-nlsf//2
        # Loop over orders
        for o in range(self.norder):
            lsf1 = ghlsf(xlsf,np.arange(self.npix),self.pars[:,o])
            lsf1[lsf1<0.] = 0.
            lsf1 /= np.tile(np.sum(lsf1,axis=1),(nlsf,1)).T
            lsf[:,:,o] = lsf1
            
        # if only one order then reshape
        if self.ndim==1: lsf=lsf.reshape(self.npix,nlsf)
            
        self._array = lsf   # save for next time

        return lsf

    
    # Return the LSF values for specific locations
    def anyarray(self,x,xtype='pixels',nlsf=15,order=0):
        """
        Return the LSF of the spectrum at specific locations.

        Unlike for the GaussianLsf class, the GaussHermiteLsf
        anyarray() method can currently only return the LSF
        at specific locations and NOT on a completely new
        wavelength array.

        Parameters
        ----------
        x : array
          The array of x-values for which to return the LSF.
        xtype : string, optional
          The type of x-values input, either 'wave' or 'pixels'.  The default
          is 'pixels'.
        order : int, optional
           The order for which to retrn the LSF array.  The default is 0.
        nlsf : int, optional
           The number of pixels to use the LSF dimension.  The default is 15.

        Returns
        -------
        lsf : array
           The 2D array of LSF values.

        Examples
        --------
        lsf = lsf.anyarray([100,200])

        """
        
        nx = len(x)
        # Only pixels supported, convert wavelength to pixels
        if xtype.lower().find('wave') > -1:
            w = x.copy()
            x = self.wave2pix(w,order=order)
        # Make LSF array
        xlsf = np.arange(nlsf)-nlsf//2
        lsf = ghlsf(xlsf,x,self.pars[:,order])
        lsf[lsf<0.] = 0.
        lsf /= np.tile(np.sum(lsf,axis=1),(nlsf,1)).T
        
        return lsf
