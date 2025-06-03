#!/usr/bin/env python

"""LSF.PY - LSF module, classes and functions

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20210605'  # yyyymmdd                                                                                                                           

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
try:
    import __builtin__ as builtins # Python 2
except ImportError:
    import builtins # Python 3

_SQRTTWO = np.sqrt(2.)

# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

cspeed = 2.99792458e5  # speed of light in km/s

# Get print function to be used locally, allows for easy logging
print = utils.getprintfunc() 

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

    Example
    -------
    .. code-block:: python

         lsf = ghlsf(x,xcenter,params)

    2015-02-26 - Written based on Nidever's code in apogeereduce - Bovy (IAS)
    Heavily modified by D.Nidever to work with Nidever's translated IDL routines Jan 2020.
    """
    nxcenter = dln.size(xcenter)
    # Parse x
    if x.ndim==1:
        x = np.tile(x,(nxcenter,1))
    # Unpack the LSF parameters
    if type(params) is not dict:
        params = unpack_ghlsf_params(params)
    # Get the wing parameters at each x
    wingparams = np.zeros((params['nWpar'],nxcenter))
    for ii in range(params['nWpar']):
        poly = np.polynomial.Polynomial(params['Wcoefs'][ii])       
        wingparams[ii] = poly(xcenter+params['Xoffset'])
    # Get the GH parameters at each x
    ghparams = np.zeros((params['Horder']+2,nxcenter))

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
    npix = nxcenter
    nlsf = x.shape[1]
    xcenter2 = np.tile(xcenter,nlsf).reshape(nlsf,npix).T
    xlsf = x + xcenter2   # absolute X-values
    # Flatten them for gausshermitebin()
    xlsf = xlsf.flatten()
    xcenter2 = xcenter2.flatten()
    # Add in height and center for gausshermitebin() and flatten to 2D
    ghparams1 = np.zeros((params['Horder']+4,nxcenter))
    ghparams1[0,:] = 1.0
    ghparams1[1,:] = xcenter
    ghparams1[2:,:] = ghparams
    ghparams2 = np.zeros((nlsf*npix,params['Horder']+4))
    for i in range(params['Horder']+4):
        ghparams2[:,i] = np.repeat(ghparams1[i,:],nlsf)
    out = gausshermitebin(xlsf,ghparams2,params['binsize'])
    
    # Calculate the Wing part of the LSF
    # Add in center for ghwingsbin() and flatten to 2D
    wingparams1 = np.zeros((params['nWpar']+1,nxcenter))
    wingparams1[0,:] = wingparams[0,:]
    wingparams1[1,:] = xcenter
    wingparams1[2:,:] = wingparams[1:,:]
    wingparams2 = np.zeros((nlsf*npix,params['nWpar']+1))
    for i in range(params['nWpar']+1):
        wingparams2[:,i] = np.repeat(wingparams1[i,:],nlsf)
    out += ghwingsbin(xlsf,wingparams2,params['binsize'],params['Wproftype'])
    
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

    Example
    -------
    .. code-block:: python

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

    # The block below is quite time-consuming, only use the Hermite parameters/columns that we need
    
    # HH are the coefficients for the powers of X
    #   hpar will be zero for orders higher than are actually desired
    if nherm==3:
        hh = np.zeros((nx,3),np.float64)      # the coefficients for the powers of X
        hh[:,0] = hpar[:,0] - hpar[:,2]/sqr2
        hh[:,1] = hpar[:,1]
        hh[:,2] = hpar[:,2]/sqr2
    elif nherm==4:
        hh = np.zeros((nx,4),np.float64)      # the coefficients for the powers of X
        hh[:,0] = hpar[:,0] - hpar[:,2]/sqr2
        hh[:,1] = hpar[:,1] - hpar[:,3]*3/np.sqrt(6)
        hh[:,2] = hpar[:,2]/sqr2
        hh[:,3] = hpar[:,3]/np.sqrt(6)
    elif nherm==5:
        hh = np.zeros((nx,5),np.float64)      # the coefficients for the powers of X
        hh[:,0] = hpar[:,0] - hpar[:,2]/sqr2 + hpar[:,4]*3/sqr24 
        hh[:,1] = hpar[:,1] - hpar[:,3]*3/np.sqrt(6)
        hh[:,2] = hpar[:,2]/sqr2 - hpar[:,4]*(6/sqr24)
        hh[:,3] = hpar[:,3]/np.sqrt(6)
        hh[:,4] = hpar[:,4]/sqr24
    elif nherm==6:
        hh = np.zeros((nx,6),np.float64)      # the coefficients for the powers of X
        hh[:,0] = hpar[:,0] - hpar[:,2]/sqr2 + hpar[:,4]*3/sqr24 
        hh[:,1] = hpar[:,1] - hpar[:,3]*3/np.sqrt(6) + hpar[:,5]*15/sqr120
        hh[:,2] = hpar[:,2]/sqr2 - hpar[:,4]*(6/sqr24)
        hh[:,3] = hpar[:,3]/np.sqrt(6) - hpar[:,5]*10/sqr120
        hh[:,4] = hpar[:,4]/sqr24
        hh[:,5] = hpar[:,5]/sqr120
    elif nherm==7:
        hh = np.zeros((nx,10),np.float64)      # the coefficients for the powers of X
        hh[:,0] = hpar[:,0] - hpar[:,2]/sqr2 + hpar[:,4]*3/sqr24 - hpar[:,6]*15/sqr720
        hh[:,1] = hpar[:,1] - hpar[:,3]*3/np.sqrt(6) + hpar[:,5]*15/sqr120
        hh[:,2] = hpar[:,2]/sqr2 - hpar[:,4]*(6/sqr24) + hpar[:,6]*45/sqr720
        hh[:,3] = hpar[:,3]/np.sqrt(6) - hpar[:,5]*10/sqr120
        hh[:,4] = hpar[:,4]/sqr24 - hpar[:,6]*15/sqr720
        hh[:,5] = hpar[:,5]/sqr120
        hh[:,6] = hpar[:,6]/sqr720
    elif nherm==8:
        hh = np.zeros((nx,10),np.float64)      # the coefficients for the powers of X
        hh[:,0] = hpar[:,0] - hpar[:,2]/sqr2 + hpar[:,4]*3/sqr24 - hpar[:,6]*15/sqr720
        hh[:,1] = hpar[:,1] - hpar[:,3]*3/np.sqrt(6) + hpar[:,5]*15/sqr120 - hpar[:,7]*105/sqr5040
        hh[:,2] = hpar[:,2]/sqr2 - hpar[:,4]*(6/sqr24) + hpar[:,6]*45/sqr720
        hh[:,3] = hpar[:,3]/np.sqrt(6) - hpar[:,5]*10/sqr120 + hpar[:,7]*105/sqr5040
        hh[:,4] = hpar[:,4]/sqr24 - hpar[:,6]*15/sqr720
        hh[:,5] = hpar[:,5]/sqr120 - hpar[:,7]*21/sqr5040
        hh[:,6] = hpar[:,6]/sqr720
        hh[:,7] = hpar[:,7]/sqr5040
    elif nherm==9:
        hh = np.zeros((nx,10),np.float64)      # the coefficients for the powers of X
        hh[:,0] = hpar[:,0] - hpar[:,2]/sqr2 + hpar[:,4]*3/sqr24 - hpar[:,6]*15/sqr720 + hpar[:,8]*105/sqr40320
        hh[:,1] = hpar[:,1] - hpar[:,3]*3/np.sqrt(6) + hpar[:,5]*15/sqr120 - hpar[:,7]*105/sqr5040
        hh[:,2] = hpar[:,2]/sqr2 - hpar[:,4]*(6/sqr24) + hpar[:,6]*45/sqr720 - hpar[:,8]*420/sqr40320
        hh[:,3] = hpar[:,3]/np.sqrt(6) - hpar[:,5]*10/sqr120 + hpar[:,7]*105/sqr5040
        hh[:,4] = hpar[:,4]/sqr24 - hpar[:,6]*15/sqr720 + hpar[:,8]*210/sqr40320
        hh[:,5] = hpar[:,5]/sqr120 - hpar[:,7]*21/sqr5040
        hh[:,6] = hpar[:,6]/sqr720 - hpar[:,8]*28/sqr40320
        hh[:,7] = hpar[:,7]/sqr5040
        hh[:,8] = hpar[:,8]/sqr40320
    else:
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

    Example
    -------
    .. code-block:: python

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
    .. code-block:: python

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
    wingparams = np.zeros((params['nWpar'],len(xcenter)))
    for ii in range(params['nWpar']):
        poly = np.polynomial.Polynomial(params['Wcoefs'][ii])       
        wingparams[ii] = poly(xcenter+params['Xoffset'])
    # Get the GH parameters at each x
    ghparams = np.zeros((params['Horder']+2,len(xcenter)))

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
    out += ghwingsbin_bovy(x,wingparams,params['binsize'],params['Wproftype'])
    
    return out


def gausshermitebin_bovy(x,params,binsize=1.0):
    """Evaluate the integrated APOGEE Gauss-Hermite function"""
    ncenter = params.shape[1]
    out = np.zeros((ncenter,x.shape[1]))
    integ = np.zeros((params.shape[0]-1,x.shape[1]))
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
    out = np.zeros((ncenter,x.shape[1]))
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
        Array of APOGEE Gauss-Hermite LSF parameters.

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
    .. code-block:: python

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

def repack_ghlsf_params(params):
    """
    Repack the APOGEE Gauss-Hermite LSF dictionary into the parameter array.

    Parameters
    ----------
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

    Returns
    -------
    lsfarr : array
        Array of APOGEE Gauss-Hermite LSF parameters.

    Examples
    --------
    .. code-block:: python

         lsfarr = repack_ghlsf_params(params)

    """
    
    # If 2D then iterate over orders and return list
    if type(params) is list:
        out = []
        for p in params:
            out.append(repack_ghlsf_params(p))
        out = np.array(out).T
        return out

    # Process single chip parameter dictionary
    lsfarr = np.array([],float)
    # binsize: The width of a pixel in X-units
    lsfarr = np.append(lsfarr,params['binsize'])
    # X0: An additive x-offset; used for GH parameters that vary globally
    lsfarr = np.append(lsfarr,params['Xoffset'])
    # Horder: The highest Hermite order
    lsfarr = np.append(lsfarr,params['Horder'])
    # Porder: Polynomial order array for global variation of each LSF parameter
    lsfarr = np.append(lsfarr,params['Porder'])
    # GHcoefs: Polynomial coefficients for sigma and the Horder Hermite parameters
    for i in range(len(params['Porder'])):
        ghcoef = params['GHcoefs'][i,:]
        lsfarr = np.append(lsfarr,ghcoef[:params['Porder'][i]+1])
    # Wproftype: Wing profile type
    lsfarr = np.append(lsfarr,params['Wproftype'])
    # nWpar: Number of wing parameters
    lsfarr = np.append(lsfarr,params['nWpar'])
    # WPorder: Polynomial order for the global variation of each wing parameter
    lsfarr = np.append(lsfarr,params['WPorder'])
    # Wcoefs: Polynomial coefficients for the wings parameters
    for i in range(len(params['WPorder'])):
        wcoef = params['Wcoefs'][i,:]
        lsfarr = np.append(lsfarr,wcoef[:params['WPorder'][i]+1])
    return lsfarr

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
          For Gaussian type this should be the polynomial coefficients for Gaussian sigma.
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
        """ Initialize Lsf object. """
        # xtype is wave or pixels.  designates what units to use BOTH for the input
        #   arrays to use with PARS and the output units
        if wave is None and xtype=='Wave':
            raise Exception('Need wavelength information if xtype=Wave')
        self.wave = wave
        if wave.ndim==1:
            npix = len(wave)
            norder = 1
        else:
            npix,norder = wave.shape        
        # Get number of pixels per order
        if self.norder>1 or wave.ndim==2:
            numpix = np.zeros(norder,int)
            for i in range(norder):
                gdpix, = np.where((wave[:,i]>0) & np.isfinite(wave[:,i]))
                if len(gdpix)>0:
                    numpix[i] = np.max(gdpix)+1
                else:
                    numpix[i] = 0
            self.numpix = numpix
        else:
            gdpix, = np.where((wave>0) & np.isfinite(wave))
            if len(gdpix)>0:
                numpix = np.max(gdpix)+1
            else:
                numpix = 0
            self.numpix = [numpix]
        # Make sure pars are 2D
        self.pars = pars
        if pars is not None:
            if np.array(pars).ndim==1:
                self.pars = np.atleast_2d(pars).T   # 2D with order dimension at 2nd
            else:
                self.pars = np.array(pars)
        self.lsftype = lsftype
        self.xtype = xtype
        # Make sure sigma is 2D
        if sigma is not None:
            if sigma.ndim==1:
                self._sigma = np.atleast_2d(sigma).T  # 2D with order dimension at 2nd
            else:
                self._sigma = sigma
        else:
            self._sigma = sigma
        self._array = None
        if (pars is None) & (sigma is None):
            if verbose is True: print('No LSF information input.  Assuming Nyquist sampling.')
            # constant FWHM=2.5, sigma=2.5/2.35
            self.pars = np.zeros((1,self.norder),float) + (2.5 / 2.3)
            self.xtype = 'Pixels'

    def __call__(self):
        """ Returns the Gaussian array.  Must be defined by the subclass."""
        pass

    def __getitem__(self,index):
        """ Return a slice or subset of the LSF object."""
        pass

    @property
    def size(self):
        """ Return number of total 'good' pixels (not including buffer pixels)."""
        return np.sum(self.numpix)

    @property
    def ndim(self):
        """ Return the number of dimensions."""
        return self.wave.ndim

    @property
    def npix(self):
        """ Return the number of pixels in each order."""
        return self.shape[0]
    
    @property
    def norder(self):
        """ Return the number of orders."""
        if self.ndim==1:
            return 1
        else:
            return self.shape[1]

    def __len__(self):
        return self.norder

    @property
    def shape(self):
        return self.wave.shape
    
    @property
    def wrange(self):
        """ Wavelength range."""
        wr = [np.inf,-np.inf]
        if self.wave.ndim==1:
            w = self.wave[0:self.numpix[0]]
            wr = [np.min(w),np.max(w)]
        else:
            for i in range(self.norder):
                w = self.wave[0:self.numpix[i],i]
                wr[0] = np.min([wr[0],np.min(w)])
                wr[1] = np.max([wr[1],np.max(w)])            
        return wr
        
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

        Example
        -------
        .. code-block:: python

             x = lsf.wave2pix(w)

        """
               
        if self.wave is None:
            raise Exception("No wavelength information")
        if self.ndim==2:
            # Order is always the second dimension
            wave = self.wave[:,order]
        else:
            wave = self.wave
        gdwave, = np.where((wave>0) & np.isfinite(wave))
        return utils.w2p(wave[gdwave],w,extrapolate=extrapolate)
        
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

        Example
        -------
        .. code-block:: python

             w = lsf.pix2wave(x)

        """
        
        if self.wave is None:
            raise Exception("No wavelength information")
        return utils.p2w(self[order].wave,x,extrapolate=extrapolate)

    def dispersion(self,x=None,xtype='pixels',extrapolate=True,order=0):
        """
        Wavelength dispersion or step (delta wavelength) at a given pixel
        or wavelength.  This is *always* positive.

        Parameters
        ----------
        x : array
           Array of pixel values at which to return the dispersion
        xtype : str, optional
           The type of x-value in put (either 'wave' or 'pixels').  The default
           is 'pixels'.
        extrapolate : bool, optional
           Extrapolate beyond the boundaries of the wavelength solution,
           if necessary.  True by default.
        order : int, optional
           The order to use if there are multiple orders.
           The default is 0.
              
        Returns
        -------
        dw : array
           The array of wavelengths dispersion/gradient values.

        Example
        -------
        .. code-block:: python

             dw = lsf.dispersion(x)

        """
        
        if self.wave is None:
            raise Exception("No wavelength information")
        wave = self[order].wave
        dwave = np.abs(np.gradient(wave))
        if x is None:
            return dwave
        if xtype.lower().find('pix')>-1:
            abscissa = np.arange(len(wave))
        else:
            abscissa = x
        dw = dln.interp(abscissa,dwave,x,kind='quadratic',assume_sorted=False,
                        extrapolate=extrapolate,exporder=1)
        if np.array(x).ndim==0: dw=dw[0]
        # fix negative values when extrapolating
        if extrapolate:
            bad = (dw < 0)
            if np.sum(bad)>0:
                # set to values on the ends
                bdlo, = np.where(x < abscissa[0])
                if len(bdlo)>0:
                    dw[bdlo] = dwave[0]
                bdhi, = np.where(x > abscissa[-1])
                if len(bdhi)>0:
                    dw[bdhi] = dwave[-1]
        return dw

    
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

        Example
        -------
        .. code-block:: python

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

    def remove_order(self,order):
        """ Remove orders from spectrum."""
        for c in ['wave','pars','_sigma']:
            if hasattr(self,c) and getattr(self,c) is not None:
                setattr(self,c,np.delete(getattr(self,c),order,axis=1))
        self.numpix = np.delete(self.numpix,order)
        # Single order, use 1-D arrays
        if self.norder==1:
            for c in ['wave','pars','_sigma']:
                if hasattr(self,c) and getattr(self,c) is not None:
                    setattr(self,c,getattr(self,c).flatten())

    def trim(self):
        """ Trim the LSF in wavelength."""
        pass

    def rebin(self,nbin):
        """ Rebin the data."""
        pass
    
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
          These are polynomial coefficients for Gaussian sigma.  If xtype='pixel', then PARS should
          be used with X pixel values and return the Gaussian sigma values in pixels.  If xtype='wave',
          then PARS should be used with wavelength in A and return the Gaussian sigma values in A as well.
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
        """ Initialize GaussianLsf object."""
        # xtype is wave or pixels.  designates what units to use BOTH for the input
        #   arrays to use with PARS and the output units
        if wave is None and xtype=='Wave':
            raise Exception('Need wavelength information if xtype=Wave')
        super().__init__(wave=wave,pars=pars,xtype=xtype,lsftype='Gaussian',sigma=sigma,verbose=False)        

        
    def __call__(self,xcen=None,x=None,xtype='pixels',nlsf=15,order=0):
        """
        Create LSF.

        Parameters
        ----------
        xcenter : int or array
           Position of the LSF center (in units specified by xtype).
        x : array
           Array of X values for which to compute the LSF (in pixel offset relative to xcenter;
             the LSF is calculated at the x offsets for each xcenter if x is 1D,
             otherwise x has to be [nxcenter,nx])).
        xtype : string, optional
           The type of x-value input, either 'wave' or 'pixels'.  Default is 'pixels'.
        order : int, optional
            The order to use if there are multiple orders.
            The default is 0.

        Returns
        -------
        lsf : array
           The LSF array.

        Example
        -------

        lsfarr = mylsf([1000.0],np.arange(15)-7)

        """

        if xcen is None:
            xcen = self.npix//2
        xcen = np.atleast_1d(xcen)
        nxcen = xcen.size
        
        if x is None:
            x = np.arange(nlsf)-nlsf//2
        else:
            if x.ndim==1:
                nlsf = len(x)
            else:
                nlsf = x.shape[1]
        
        # Get wavelength and pixel arrays
        if xtype.lower().find('pix') > -1:
            wcen = self.pix2wave(xcen,order=order)
            w = self.pix2wave(x,order=order)
        else:
            wcen = xcen.copy()
            xcen = self.wave2pix(wcen,order=order)
            w = x.copy()
            x = self.wave2pix(w,order=order)
        
        # returns xsigma in units of self.xtype not necessarily xtype
        xsigma = self.sigma(xcen,xtype=xtype,order=order)

        # Convert sigma from wavelength to pixels, if necessary
        if self.xtype.lower().find('wave') > -1:
            wsigma = xsigma.copy()
            w1 = self.pix2wave(np.array(xcen)+1,order=order)
            dw = np.abs(w1-w)
            xsigma = wsigma / dw

        # Make LSF array
        if x.ndim==2:  # [Nxcenter,Nx]
            xsigma2 = np.repeat(xsigma,nlsf).reshape((nxcen,nlsf))
            lsf = np.exp(-0.5*x**2 / xsigma2**2) / (np.sqrt(2*np.pi)*xsigma2)            
        else:
            xlsf2 = np.repeat(x,nxcen).reshape((nlsf,nxcen)).T
            xsigma2 = np.repeat(xsigma,nlsf).reshape((nxcen,nlsf))
            lsf = np.exp(-0.5*xlsf2**2 / xsigma2**2) / (np.sqrt(2*np.pi)*xsigma2)
        lsf[lsf<0.] = 0.
        if lsf.ndim==2:
            lsf /= np.tile(np.sum(lsf,axis=1),(nlsf,1)).T
        else:
            lsf /= np.sum(lsf)
            
        return lsf
    
        
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
           The array of Gaussian sigma values.  The sigma values will
           *always* be in units of lsf.xtype.

        Example
        -------
        .. code-block:: python

             sigma = lsf.sigma([100,200])
        
        """
        
        # The sigma will be returned in units given in lsf.xtype
        if self._sigma is not None:
            _sigma = self._sigma
            if self._sigma.ndim==2: _sigma = self._sigma[:,order]
            if x is None:
                return _sigma
            else:
                # Wavelength input
                if xtype.lower().find('wave') > -1:
                    x0 = np.array(x).copy()            # backup
                    x = self.wave2pix(x0,order=order)  # convert to pixels
                    
                # Integer, just return the values
                if (type(x)==int) | (np.array(x).dtype.kind=='i'):
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
                            coef1 = dln.poly_fit(xin[0:10], _sigma[0:10], 1)
                            bd1, nbd1 = dln.where(x <0)
                            sig[bd1] = dln.poly(x[bd1],coef1)
                        # At the end
                        if (np.max(x)>(npix-1)):
                            coef2 = dln.poly_fit(xin[npix-10:], _sigma[npix-10:], 1)
                            bd2, nbd2 = dln.where(x > (npix-1))
                            sig[bd2] = dln.poly(x[bd2],coef2)

                    return sig
                        
        # Need to calculate sigma from pars
        else:
            if x is None:
                x = np.arange(self.npix)
                xtype = 'Pixels'
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
            if smlen==0: smlen=51
            _sigma = self._sigma.reshape(self.npix,self.norder)   # make 2D
            for o in range(self.norder):
                sig = _sigma[:,o]
                gd,ngd,bd,nbd = dln.where(sig > 0.001,comp=True)
                if nbd>0:
                    sig[bd] = np.nan
                    smsig = dln.gsmooth(sig,smlen)
                    sig[bd] = smsig[bd]
                gd,ngd,bd,nbd = dln.where(sig > 0.001,comp=True)
                if nbd>0:
                    sig[bd] = np.nan
                    smsig = dln.gsmooth(sig,3*smlen)
                    sig[bd] = smsig[bd]
                gd,ngd,bd,nbd = dln.where(sig > 0.001,comp=True)
                if nbd>0:
                    sig[bd] = np.median(sig[gd])                    
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

        Example
        -------
        .. code-block:: python

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
            nlsf = int(np.round(np.max(xsigma)*6))
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
           The order for which to return the LSF array.  The default is 0.
        original : bool, optional
           If original=True, then the LSFs are returned on the original
           wavelength scale but at the centers given in "x".
           If the LSF is desired on a completely new wavelength scale
           (given by "x"), then original=False should be used instead.

        Returns
        -------
        lsf : array
           The 2D array of LSF values.

        Example
        -------
        .. code-block:: python

             lsf = lsf.anyarray([100,200])

        """

        nx = len(x)
        # returns xsigma in units of self.xtype (NOTE: not necessarily *input* xtype)
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

        # Convert sigma from original pixels to new
        #  wavelength/pixel scale, if necessary
        else:
            # New wavelength/pixel scale
            if original==False:
                origdw = self.dispersion(x,xtype='pixels',order=order)
                #w1 = self.pix2wave(np.array(x)+1,order=order)
                #origdw = np.abs(w1-w)
                newdw = np.abs(np.gradient(w))
                xsigma *= origdw / newdw
            
        # Figure out nLSF pixels needed, +/-3 sigma
        nlsf = int(np.round(np.max(xsigma)*6))
        if nlsf % 2 == 0: nlsf+=1                   # must be odd
        
        # Make LSF array
        lsf = np.zeros((nx,nlsf))
        xlsf = np.arange(nlsf)-nlsf//2
        xlsf2 = np.repeat(xlsf,nx).reshape((nlsf,nx)).T
        xsigma2 = np.repeat(xsigma,nlsf).reshape((nx,nlsf))
        lsf = np.exp(-0.5*xlsf2**2 / xsigma2**2) / (np.sqrt(2*np.pi)*xsigma2)
        lsf[lsf<0.] = 0.
        if lsf.ndim==2:
            lsf /= np.tile(np.sum(lsf,axis=1),(nlsf,1)).T
        else:
            lsf /= np.sum(lsf)
            
        # should I use gaussbin????
        return lsf

    def trim(self,w0,w1):
        """
        Trim the LSF in wavelength.

        Parameters
        ----------
        w0 : float
           Lower wavelength (in Ang) to trim.
        w1 : float
           Upper wavelength (in Ang) to trim.

        Returns
        -------
        Nothing is returned.  The spectrum is trimmed in place.

        Examples
        --------

        lsf.trim(5500.0,6500.0)

        """
        # Get sigma, we might need this below
        if self.xtype.lower().find('pix')>-1:
            sigma = np.zeros([self.npix,self.norder],float)
            for o in range(self.norder):
                x = np.arange(self.numpix[o])
                sigma[:self.numpix[o],o] = self.sigma(x,order=o)
        else:
            sigma = None
        # Get number of pixels per order
        ranges = np.zeros([self.norder,2],int)-1
        ngood = np.zeros(self.norder,int)        
        for o in range(self.norder):
            if self.norder==1:
                ind, = np.where((self.wave>=w0) & (self.wave<=w1))
            else:
                npix = self.numpix[o]
                ind, = np.where((self.wave[:npix,o]>=w0) & (self.wave[:npix,o]<=w1))
            ngood[o] = len(ind)
            if len(ind)>0:
                ranges[o,0] = ind[0]
                ranges[o,1] = ind[-1]
        # Check that we have some pixels
        goodorder, = np.where(ngood>0)
        if len(goodorder)==0:
            raise ValueError('No pixels left')
        # Remove blank orders
        blankorder, = np.where(ngood==0)
        for o in blankorder:
            self.remove_order(o)
        if sigma is not None:
            sigma = sigma[:,goodorder]
            if len(goodorder)==1:
                sigma = sigma.flatten()
        ranges = ranges[goodorder,:]
        numpix = list(ranges[:,1]-ranges[:,0]+1)
        npix = np.max(numpix)    # new npix  
        if len(goodorder)==1:
            ranges = ranges.flatten()
        # Trim arrays
        default = {'wave':0.0,'_sigma':0.0}
        for c in ['wave','_sigma']:
            if hasattr(self,c) and getattr(self,c) is not None:
                arr = getattr(self,c)
                if self.norder==1:
                    newarr = arr[ranges[0]:ranges[1]+1]
                else:
                    newarr = np.zeros([npix,self.norder],arr.dtype)
                    newarr[:,:] = default[c]
                    for o in range(self.norder):
                        newvals = arr[ranges[o,0]:ranges[o,1]+1,o].copy()
                        newarr[:len(newvals),o] = newvals
                setattr(self,c,newarr)
        self.numpix = numpix
        if sigma is not None:
            if self.norder==1:
                sigma = sigma[ranges[0]:ranges[1]+1]
            else:
                for o in range(self.norder):
                    newvals = sigma[ranges[o,0]:ranges[o,1]+1,o].copy()
                    sigma[:len(newvals),o] = newvals
        # Fix pars if using pixel-based values
        #   xtype='Wave' is fine
        if self.xtype.lower().find('pix')>-1 and hasattr(self,'pars') and self.pars is not None:
            for o in range(self.norder):
                x = np.arange(self.numpix[o])
                if self.norder==1:
                    sigma1 = sigma[:self.numpix[o]]
                else:
                    sigma1 = sigma[:self.numpix[o],o]
                coef = np.polyfit(x,sigma1,self.pars.shape[0]-1)
                self.pars[:,o] = coef[::-1]

    def rebin(self,nbin,mean=False):
        """
        Rebin the LSF information in the spectral dimension.

        Parameters
        ----------
        nbin : int
            Integer number of pixels to bin.
        mean : bool, optional
            Take the mean instead of the default sum.

        Returns
        -------
        Nothing is returned.  The spectrum is rebinned in place.

        Examples
        --------

        lsf.rebin(2)

        """
        # Only keep complete bins and remove any excess
        newnpix = self.npix//nbin
        numpix = self.numpix.copy()        
        # Bin the data
        default = {'wave':0.0,'_sigma':0.0}
        for c in ['wave','_sigma']:
            if hasattr(self,c) and getattr(self,c) is not None:
                arr = getattr(self,c)
                if self.norder==1:
                    arr = np.atleast_2d(arr).T
                # Use mean for both wave and _sigma
                newarr = np.zeros([newnpix,self.norder],arr.dtype)
                newarr[:,:] = default[c]
                for o in range(self.norder):
                    arr1 = arr[:self.numpix[o],o]  # trim to good values
                    newvals = dln.rebin(arr1,binsize=nbin)
                    if c=='_sigma' and self.xtype.lower().find('pix')>-1:
                        newvals /= nbin    # reduce for new binsize
                    newarr[:len(newvals),o] = newvals
                    numpix[o] = len(newvals)
                if self.norder==1:
                    newarr = newarr.flatten()
                setattr(self,c,newarr)
        self.numpix = numpix
        # Fix pars if using pixel-based values
        #   xtype='Wave' is fine
        if self.xtype.lower().find('pix')>-1 and hasattr(self,'pars') and self.pars is not None:
            # Since we are going from X -> X/nbin
            # we need to scale the polynomial coefficients
            # c0 -> c0
            # c1 -> c1*nbin
            # c2 -> c2*nbin**2
            for i in range(self.pars.shape[0]):
                self.pars[i,:] *= nbin**i
            # Reduce sigma since our pixels are getting larget
            self.pars /= nbin
        
    def __getitem__(self,index):
        """ 
        Return slice or subset of LSF object.

        This returns a new GaussianLSF with arrays/attributes that are passed by REFERENCE.
        This way you can modify the values in the returned object
        and it will modify the original arrays.

        There are four types of inputs
        1) integer - returns a specific order
        2) 1-D slice (single order) - return that slice of the data
        3) 1-D slice (multiple orders) - return multiple orders using slice
        4) 1-D slice + integer - returns slice of data for order index
        """

        # Single index requested
        if isinstance(index,int):
            case = 1
            order = index
            norder = 1
            slc = slice(0,self.numpix[order])
            if index>=self.norder:
                raise IndexError('Index is too large')
        # Slice
        elif isinstance(index,slice):
            # Single order
            if self.norder==1:
                case = 2
                order = 0
                norder = 1
                slc = np.arange(self.numpix[order])[index]
                #   this ensures that the slice returns the minimum size                
            # Multiple orders
            else:
                case = 3
                order = index
                oindex = np.arange(self.norder)[order]
                norder = len(oindex)
                slc = slice(None,None,None)  # all
                if norder==0:
                    raise IndexError('Order index out of bounds for size '+str(self.norder))
                # Want all of the orders
                if norder==self.norder:
                    return self
        # Slice + index
        elif isinstance(index,tuple):
            case = 4
            if isinstance(index[0],slice)==False or isinstance(index[1],int)==False:
                raise IndexError('Must be slice + order index')
            order = index[1]
            slc = np.arange(self.numpix[order])[index[0]]
            #   this ensures that the slice returns the minimum size
            norder = 1
        else:
            raise IndexError('Index must be an integer, slice, or slice+integer')

        kwargs = {}
        if hasattr(self,'wave') and self.wave is not None:
            if self.norder==1:
                kwargs['wave'] = self.wave[slc]
            else:
                kwargs['wave'] = self.wave[slc,order]
            if norder==1: kwargs['wave'] = kwargs['wave'].flatten()
        if hasattr(self,'pars') and self.pars is not None:
            if self.norder==1:
                if np.array(self.pars).ndim==2:
                    kwargs['pars'] = self.pars[:,0]
                else:
                    kwargs['pars'] = self.pars
            else:
                kwargs['pars'] = self.pars[:,order]
        # Scalars
        for c in ['xtype','lsftype']:
            kwargs[c] = getattr(self,c)
        # Make the LSF   
        olsf = GaussianLsf(**kwargs)
        # Adding _sigma
        if hasattr(self,'_sigma') and self._sigma is not None:
            if self.norder==1:
                olsf._sigma = self._sigma[slc]
            else:
                olsf._sigma = self._sigma[slc,order]
            if norder==1: olsf._sigma = olsf._sigma.flatten()

        # Single-order pixel slicing, need to fix pars
        #   only if xtype=pixels and we have pars
        if (case==2 or case==4) and self.xtype.lower().find('pix')>-1 and hasattr(self,'pars'):
            sigma1 = self.sigma(order=order)[slc]
            x = np.arange(olsf.npix)
            coef = np.polyfit(x,sigma1,len(self.pars.shape[:,0])-1)
            olsf.pars[:,0] = coef[::-1]
            
        return olsf

                
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
        """ Initialize GaussianHermiteLsf object."""
        # xtype is wave or pixels.  designates what units to use BOTH for the input
        #   arrays to use with PARS and the output units
        if wave is None and xtype=='Wave':
            raise Exception('Need wavelength information if xtype=Wave')
        super().__init__(wave=wave,pars=pars,xtype='pixel',lsftype='Gauss-Hermite',sigma=sigma,verbose=False)

        
    def __call__(self,xcen=None,x=None,xtype='pixels',nlsf=15,order=0):
        """
        Create LSF.

        Parameters
        ----------
        xcenter : int or array
           Position of the LSF center (in units specified by xtype).
        x : array
           Array of X values for which to compute the LSF (in pixel offset relative to xcenter;
             the LSF is calculated at the x offsets for each xcenter if x is 1D,
             otherwise x has to be [nxcenter,nx])).
        xtype : string, optional
           The type of x-value input, either 'wave' or 'pixels'.  Default is 'pixels'.
        order : int, optional
            The order to use if there are multiple orders.
            The default is 0.

        Returns
        -------
        lsf : array
           The LSF array.

        Example
        -------

        lsfarr = mylsf([1000.0],np.arange(15)-7)

        """

        # Wavelengths input
        if xtype.lower().find('wave') > -1:
            w = x.copy()
            x = self.wave2pix(w,order=order)
            wcen = xcen.copy()
            xcen = self.wave2pix(wcen,order=order)            
            
        if xcen is None:
            xcen = self.npix//2
        xcen = np.atleast_1d(xcen)
            
        if x is None:
            x = np.arange(nlsf)-nlsf//2
        else:
            if x.ndim==1:
                nlsf = len(x)
            else:
                nlsf = x.shape[1]
            
        # Make the LSF using ghlsf()
        lsf = ghlsf(x,xcen,self.pars[:,order])
        
        # Make sure it's normalized
        lsf[lsf<0.] = 0.
        if lsf.ndim==2:
            lsf /= np.tile(np.sum(lsf,axis=1),(nlsf,1)).T
        else:
            lsf /= np.sum(lsf)
            
        return lsf

    
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

        Example
        -------
        .. code-block:: python

             sigma = lsf.sigma([100,200])
        
        """
        
        # The sigma will be returned in pixels
        if x is None:
            x = np.arange(self.npix)
        if self.pars is None:
            raise Exception("No LSF parameters")
        # Get parameters
        pars = self.pars
        if pars.ndim==2: pars=self.pars[:,order]
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

        Example
        -------
        .. code-block:: python

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
    def anyarray(self,x,xtype='pixels',nlsf=15,order=0,original=True,**kwargs):
        """
        Return the LSF of the spectrum at specific locations.

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
        original : bool, optional
           If original=True, then the LSFs are returned on the original
           wavelength scale but at the centers given in "x".
           If the LSF is desired on a completely new wavelength scale
           (given by "x"), then orignal=False should be used instead.

        Returns
        -------
        lsf : array
           The 2D array of LSF values.

        Example
        -------
        .. code-block:: python
        
             lsf = lsf.anyarray([100,200])

        """
        
        x = np.atleast_1d(x)
        # Make sure nLSF is odd
        if nlsf % 2 == 0: nlsf+=1
        
        nx = len(x)
        # Gauss-Hermite xtype is always in pixels, convert wavelength to pixels
        if xtype.lower().find('wave') > -1:
            w = x.copy()
            x = self.wave2pix(w,order=order)
        
        # Make xLSF array
        
        # New pixel scale
        if original==False:
            dx = x[1:]-x[0:-1]
            dx = np.hstack((dx,dx[-1]))
            xlsf1 = np.arange(nlsf)-nlsf//2
            xlsf = np.outer(dx,xlsf1)
        # Original pixel scale
        else:
            xlsf = np.arange(nlsf)-nlsf//2
            
        # Make the LSF using ghlsf()
        lsf = ghlsf(xlsf,x,self.pars[:,order])
        
        # Make sure it's normalized
        lsf[lsf<0.] = 0.
        if lsf.ndim==2:
            lsf /= np.tile(np.sum(lsf,axis=1),(nlsf,1)).T
        else:
            lsf /= np.sum(lsf)

        # if original=False, then this normalization might not be
        # right, need to multiply by dx.
            
        return lsf

    def trim(self,w0,w1):
        """
        Trim the LSF in wavelength.

        Parameters
        ----------
        w0 : float
           Lower wavelength (in Ang) to trim.
        w1 : float
           Upper wavelength (in Ang) to trim.

        Returns
        -------
        Nothing is returned.  The spectrum is trimmed in place.

        Examples
        --------

        lsf.trim(5500.0,6500.0)

        """
        # Get number of pixels per order
        ranges = np.zeros([self.norder,2],int)-1
        ngood = np.zeros(self.norder,int)        
        for o in range(self.norder):
            if self.norder==1:
                ind, = np.where((self.wave>=w0) & (self.wave<=w1))
            else:
                npix = self.numpix[o]
                ind, = np.where((self.wave[:npix,o]>=w0) & (self.wave[:npix,o]<=w1))
            ngood[o] = len(ind)
            if len(ind)>0:
                ranges[o,0] = ind[0]
                ranges[o,1] = ind[-1]
        # Check that we have some pixels
        goodorder, = np.where(ngood>0)
        if len(goodorder)==0:
            raise ValueError('No pixels left')
        # Remove blank orders
        blankorder, = np.where(ngood==0)
        for o in blankorder:
            self.remove_order(o)
        ranges = ranges[goodorder,:]
        numpix = list(ranges[:,1]-ranges[:,0]+1)
        npix = np.max(numpix)    # new npix  
        if len(goodorder)==1:
            ranges = ranges.flatten()
        # Trim arrays
        default = {'wave':0.0,'_sigma':0.0}
        for c in ['wave','_sigma']:
            if hasattr(self,c) and getattr(self,c) is not None:
                arr = getattr(self,c)
                if self.norder==1:
                    newarr = arr[ranges[0]:ranges[1]+1]
                else:
                    newarr = np.zeros([npix,self.norder],arr.dtype)
                    newarr[:,:] = default[c]
                    for o in range(self.norder):
                        newvals = arr[ranges[o,0]:ranges[o,1]+1,o].copy()
                        newarr[:len(newvals),o] = newvals
                setattr(self,c,newarr)
        self.numpix = numpix
        # Fix pars if using pixel-based values
        #   xtype='Wave' is fine
        if self.xtype.lower().find('pix')>-1:
            # Just shift the Xshift parameter
            if self.norder==1:
                self.pars[1,0] += ranges[0]
            else:
                for o in range(self.norder):
                    self.pars[1,o] += ranges[o,0]

    def rebin(self,nbin,mean=False):
        """
        Rebin the LSF information in the spectral dimension.

        Parameters
        ----------
        nbin : int
            Integer number of pixels to bin.
        mean : bool, optional
            Take the mean instead of the default sum.

        Returns
        -------
        Nothing is returned.  The spectrum is rebinned in place.

        Examples
        --------

        lsf.rebin(2)

        """
        # Only keep complete bins and remove any excess
        newnpix = self.npix//nbin
        numpix = self.numpix.copy()
        oldnumpix = self.numpix.copy()
        # Bin the data
        default = {'wave':0.0,'_sigma':0.0}
        for c in ['wave','_sigma']:
            if hasattr(self,c) and getattr(self,c) is not None:
                arr = getattr(self,c)
                if self.norder==1:
                    arr = np.atleast_2d(arr).T
                # Use mean for both wave and _sigma
                newarr = np.zeros([newnpix,self.norder],arr.dtype)
                newarr[:,:] = default[c]
                for o in range(self.norder):
                    arr1 = arr[:self.numpix[o],o]  # trim to good values
                    newvals = dln.rebin(arr1,binsize=nbin)
                    if c=='_sigma' and self.xtype.lower().find('pix')>-1:
                        newvals /= nbin    # reduce for new binsize
                    newarr[:len(newvals),o] = newvals
                    numpix[o] = len(newvals)
                if self.norder==1:
                    newarr = newarr.flatten()
                setattr(self,c,newarr)
        self.numpix = numpix
        # Fix pars if using pixel-based values
        #   xtype='Wave' is fine
        if self.xtype.lower().find('pix')>-1:
            # apogee_drp apdithercomb.pro does something similar
            pardict = unpack_ghlsf_params(self.pars)            
            for o in range(self.norder):
                params = pardict[o]
                newparams = params.copy()
                newparams['binsize'] *= nbin
                newparams['Xoffset'] /= nbin
                # The Gauss-Hermite coefficients:
                # [sigma, H0, H1, H2, H3, H4, ...] up to H9.
                # Since we're changing X -> X/nbin we need to scale the
                # polynomial coefficients depending on their power
                # c0 -> c0
                # c1 -> c1*nbin
                # c2 -> c2*nbin**2   and so on
                # GHcoefs is [GH parameter, Polynomial order]
                #  skip 1st, constant them
                for i in np.arange(1,np.max(params['Porder'])+1):
                    # change all GH parameters of the ith order
                    #   if they are zero, nothing will cahnge
                    newparams['GHcoefs'][:,i] *= nbin**i
                # Reduce GH sigma (1st parameter) since our pixels are getting larger
                newparams['GHcoefs'][0,:] /= nbin
                # Same for wing parameters
                for i in np.arange(1,np.max(params['WPorder'])+1):
                    newparams['Wcoefs'][:,i] *= nbin**i
                # Reduce wing sigma (1st parameter) since our pixels are getting larger
                newparams['Wcoefs'][0,:] /= nbin
                pardict[o] = newparams
            lsfarr = repack_ghlsf_params(pardict)
            self.pars = lsfarr
                
    def __getitem__(self,index):
        """ 
        Return slice or subset of LSF object.

        This returns a new GaussHermiteLSF with arrays/attributes that are passed by REFERENCE.
        This way you can modify the values in the returned object
        and it will modify the original arrays.

        There are four types of inputs
        1) integer - returns a specific order
        2) 1-D slice (single order) - return that slice of the data
        3) 1-D slice (multiple orders) - return multiple orders using slice
        4) 1-D slice + integer - returns slice of data for order index
        """

        # Single index requested
        if isinstance(index,int):
            case = 1
            order = index
            norder = 1
            slc = slice(0,self.numpix[order])
            if index>=self.norder:
                raise IndexError('Index is too large')
        # Slice
        elif isinstance(index,slice):
            # Single order
            if self.norder==1:
                case = 2
                order = 0
                norder = 1
                slc = np.arange(self.numpix[order])[index]
                #   this ensures that the slice returns the minimum size                
            # Multiple orders
            else:
                case = 3
                order = index
                oindex = np.arange(self.norder)[order]
                norder = len(oindex)
                slc = slice(None,None,None)  # all
                if norder==0:
                    raise IndexError('Order index out of bounds for size '+str(self.norder))
                # Want all of the orders
                if norder==self.norder:
                    return self
        # Slice + index
        elif isinstance(index,tuple):
            case = 4
            if isinstance(index[0],slice)==False or isinstance(index[1],int)==False:
                raise IndexError('Must be slice + order index')
            order = index[1]
            slc = np.arange(self.numpix[order])[index[0]]
            #   this ensures that the slice returns the minimum size
            norder = 1
        else:
            raise IndexError('Index must be an integer, slice, or slice+integer')

        kwargs = {}
        if hasattr(self,'wave') and self.wave is not None:
            if self.norder==1:
                kwargs['wave'] = self.wave[slc]
            else:
                kwargs['wave'] = self.wave[slc,order]
            if norder==1: kwargs['wave'] = kwargs['wave'].flatten()
        if hasattr(self,'pars') and self.pars is not None:
            if self.norder==1:
                kwargs['pars'] = self.pars[:].copy()
            else:
                kwargs['pars'] = self.pars[:,order].copy()
        # Scalars
        for c in ['xtype','lsftype']:
            kwargs[c] = getattr(self,c)
        # Make the LSF   
        olsf = GaussHermiteLsf(**kwargs)
        # Adding _sigma
        if hasattr(self,'_sigma') and self._sigma is not None:
            if self.norder==1:
                olsf._sigma = self._sigma[slc]
            else:
                olsf._sigma = self._sigma[slc,order]
            if norder==1: olsf._sigma = olsf._sigma.flatten()
        
        # Single-order pixel slicing, need to fix pars
        #   only if xtype=pixels and we have pars
        if (case==2 or case==4) and self.xtype.lower().find('pix')>-1 and hasattr(self,'pars'):
            # Just shift the Xshift parameter
            olsf.pars[1,0] += slc[0]
            
        return olsf
