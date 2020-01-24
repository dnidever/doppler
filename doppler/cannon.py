#!/usr/bin/env python

"""CANNON.PY - Routines to work with Cannon models.

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20200112'  # yyyymmdd

import os
import numpy as np
import warnings
from glob import glob
from scipy.interpolate import interp1d
import thecannon as tc
from dlnpyutils import utils as dln, bindata
from .spec1d import Spec1D
from . import utils
import copy

# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

cspeed = 2.99792458e5  # speed of light in km/s

# astropy.modeling can handle errors and constraints

def cannon_copy(model):
    """ Make a new copy of a Cannon model."""
    npix, ntheta = model._theta.shape
    nlabels = len(model.vectorizer.label_names)
    labelled_set = np.zeros([2,nlabels])
    normalized_flux = np.zeros([2,npix])
    normalized_ivar = normalized_flux.copy()*0
    omodel = tc.CannonModel(labelled_set,normalized_flux,normalized_ivar,model.vectorizer)
    # Copy over all of the attributes
    for name, value in vars(model).items():
        if name not in ['_vectorizer']:
            setattr(omodel,name,copy.deepcopy(value))
    return omodel

def hstack(models):
    """ Stack Cannon models.  Basically combine all of the pixels right next to each other."""
    nmodels = dln.size(models)
    if nmodels==1: return models

    # Number of combined pixels
    nfpix = 0
    for i in range(nmodels): nfpix += len(models[i].dispersion)

    # Initiate final Cannon model
    npix, ntheta = models[0]._theta.shape
    nlabels = len(models[0].vectorizer.label_names)
    labelled_set = np.zeros([2,nlabels])
    normalized_flux = np.zeros([2,nfpix])
    normalized_ivar = normalized_flux.copy()*0
    omodel = tc.CannonModel(labelled_set,normalized_flux,normalized_ivar,models[0].vectorizer)
    omodel._s2 = np.empty(nfpix,np.float64)
    omodel._scales = models[0]._scales.copy()
    omodel._theta = np.empty((nfpix,ntheta),np.float4)
    omodel._design_matrix = models[0]._design_matrix.copy()
    omodel._fiducials = models[0]._fiducials.copy()
    omodel.dispersion = np.empty(nfpix,np.float64)
    omodel.regularization = models[0].regularization
    omodel.ranges = models[0].ranges

    # scales, design_matrix, fiducials should be identical or we have problems
    if np.sum((models[0]._scales!=models[1]._scales)) + np.sum((models[0]._design_matrix!=models[1]._design_matrix)) + \
       np.sum((models[0]._fiducials!=models[1]._fiducials)) + np.sum((models[0].ranges!=models[1].ranges)) > 0:
              raise ValueError('scales, design_matrix, fiducials, and ranges must be identical in the Cannon models')
    
    # Fill in the information
    off = 0  # offset
    for i in range(nmodels):
        model = models[i]
        npix, ntheta = model._theta.shape
        omodel._s2[off:off+npix,:] = model._s2
        omodel._theta[off:off+npix,:] = model._theta
        omodel.dispersion[off:off+npix] = model.dispersion
        off += npix
        
    return omodel
        

# Load the cannon model
def load_all_cannon_models():
    """
    Load all Cannon models from the Doppler data/ directory.

    Returns
    -------
    models : list of Cannon models
       List of all Cannon models in the Doppler data/ directory.

    Examples
    --------
    models = load_all_cannnon_models()

    """
    
    datadir = utils.datadir()
    files = glob(datadir+'cannongrid*.pkl')
    nfiles = len(files)
    if nfiles==0:
        raise Exception("No Cannon model files in "+datadir)
    models = []
    for f in files:
        model1 = tc.CannonModel.read(f)
        ranges = np.zeros([3,2])
        for i in range(3):
            ranges[i,:] = dln.minmax(model1._training_set_labels[:,i])
        model1.ranges = ranges
        # Rename _fwhm to fwhm and _wavevac to wavevac
        if hasattr(model1,'_fwhm'):
            setattr(model1,'fwhm',model1._fwhm)
            delattr(model1,'_fwhm')
        if hasattr(model1,'_wavevac'):
            setattr(model1,'wavevac',model1._wavevac)
            delattr(model1,'_wavevac')            
        models.append(model1)
    return models

# Get correct cannon model for a given set of inputs
def get_best_cannon_model(models,pars):
    """
    Return the Cannon model that covers the input parameters.
    It returns the first one that matches.

    Parameters
    ----------
    models : list of Cannon models
       List of Cannon models to check.
    pars : array
       Array of parameters/labels to check the Cannon models for.

    Returns
    -------
    m : Cannon model
      The Cannon model that covers the input parameters.

    Examples
    --------

    m = get_best_cannon_models(models,pars)

    """
    
    n = dln.size(models)
    if n==1:
        return models
    for m in models:
        #inside.append(m.in_convex_hull(pars))        
        # in_convex_hull() is very slow
        # Another list for multiple orders
        if dln.size(m)>1:
            ranges = m[0].ranges  # just take the first one
        else:
            ranges = m.ranges
        inside = True
        for i in range(3):
            inside &= (pars[i]>=ranges[i,0]) & (pars[i]<=ranges[i,1])
        if inside:
            return m
    
    return None

# Create a "wavelength-trimmed" version of a CannonModel model (or multiple models)
def trim_cannon_model(model,x0=None,x1=None,w0=None,w1=None):
    """
    Trim a Cannon model (or list of Cannon models) to a smaller
    wavelength range.  Either x0 and x1 must be input or w0 and w1.

    Parameters
    ----------
    model : Cannon model or list
       Cannon model or list of Cannon models.
    x0 : int, optional
      The starting pixel to trim to.
    x1 : int, optional
      The ending pixel to trim to.  Must be input with x0.
    w0 : float, optional
      The starting wavelength to trim to.
    w1 : float, optional
      The ending wavelength to trim to.  Must be input with w0.

    Returns
    -------
    omodel : Cannon model(s)
       The trimmed Cannon model(s).

    Examples
    --------

    omodel = trim_cannon_model(model,100,1000)

    """
    
    if type(model) is list:
        omodel = []
        for i in range(len(model)):
            model1 = model[i]
            omodel1 = trim_cannon_model(model1,x0=x0,x1=x1,w0=w0,w1=w1)
            omodel.append(omodel1)
    else:
        if x0 is None:
            x0 = np.argmin(np.abs(model.dispersion-w0))
        if x1 is None:
            x1 = np.argmin(np.abs(model.dispersion-w1))
        npix = x1-x0+1
        nlabels = len(model.vectorizer.label_names)
        labelled_set = np.zeros([2,nlabels])
        normalized_flux = np.zeros([2,npix])
        normalized_ivar = normalized_flux.copy()*0
        omodel = tc.CannonModel(labelled_set,normalized_flux,normalized_ivar,model.vectorizer)
        omodel._s2 = model._s2[x0:x1+1]
        omodel._scales = model._scales
        omodel._theta = model._theta[x0:x1+1,:]
        omodel._design_matrix = model._design_matrix
        omodel._fiducials = model._fiducials
        omodel.dispersion = model.dispersion[x0:x1+1]
        omodel.regularization = model.regularization
        if hasattr(model,'ranges') is True: omodel.ranges=model.ranges
        
    return omodel


# Rebin a CannonModel model
def rebin_cannon_model(model,binsize):
    """
    Rebin a Cannon model (or list of models) by an integer amount.

    Parameters
    ----------
    model : Cannon model or list
       The Cannon model or list of them to rebin.
    binsize : int
       The number of pixels to bin together.

    Returns
    -------
    omodel : Cannon model or list
       The rebinned Cannon model or list of Cannon models.

    Examples
    --------

    omodel = rebin_cannon_model(model,4

    """
    
    if type(model) is list:
        omodel = []
        for i in range(len(model)):
            model1 = model[i]
            omodel1 = rebin_cannon_model(model1,binsize)
            omodel.append(omodel1)
    else:
        npix, npars = model.theta.shape
        npix2 = np.round(npix // binsize).astype(int)
        nlabels = len(model.vectorizer.label_names)
        labelled_set = np.zeros([2,nlabels])
        normalized_flux = np.zeros([2,npix2])
        normalized_ivar = normalized_flux.copy()*0
        omodel = tc.CannonModel(labelled_set,normalized_flux,normalized_ivar,model.vectorizer)
        omodel._s2 = dln.rebin(model._s2[0:npix2*binsize],npix2)
        omodel._scales = model._scales
        omodel._theta = np.zeros((npix2,npars),np.float64)
        for i in range(npars):
            omodel._theta[:,i] = dln.rebin(model._theta[0:npix2*binsize,i],npix2)
        omodel._design_matrix = model._design_matrix
        omodel._fiducials = model._fiducials
        if model.dispersion is not None:
            omodel.dispersion = dln.rebin(model.dispersion[0:npix2*binsize],npix2)
        omodel.regularization = model.regularization
        if hasattr(model,'ranges') is True: omodel.ranges=model.ranges
        
    return omodel


# Interpolate a CannonModel model
def interp_cannon_model(model,xout=None,wout=None):
    """
    Interpolate a Cannon model or list of models onto a new wavelength
    (or pixel) scale.  Either xout or wout must be input.

    Parameters
    ----------
    model : Cannon model or list
      Cannon model or list of Cannon models to interpolate.
    xout : array, optional
      The desired output pixel array.
    wout : array, optional
      The desired output wavelength aray.

    Returns
    -------
    omodel : Cannon model or list
      The interpolated Cannon model or list of models.

    Examples
    --------

    omodel = interp_cannon_model(model,wout)

    """

    if (xout is None) & (wout is None):
        raise Exception('xout or wout must be input')
    
    if type(model) is list:
        omodel = []
        for i in range(len(model)):
            model1 = model[i]
            omodel1 = interp_cannon_model(model1,xout=xout,wout=wout)
            omodel.append(omodel1)
    else:

        if (wout is not None) & (model.dispersion is None):
            raise Exception('wout input but no dispersion information in model')
    
        # Convert wout to xout
        if (xout is None) & (wout is not None):
            npix, npars = model.theta.shape
            x = np.arange(npix)
            xout = interp1d(model.dispersion,x,kind='cubic',bounds_error=False,
                            fill_value=(np.nan,np.nan),assume_sorted=True)(wout)
        npix, npars = model.theta.shape
        npix2 = len(xout)
        nlabels = len(model.vectorizer.label_names)
        labelled_set = np.zeros([2,nlabels])
        normalized_flux = np.zeros([2,npix2])
        normalized_ivar = normalized_flux.copy()*0
        omodel = tc.CannonModel(labelled_set,normalized_flux,normalized_ivar,model.vectorizer)
        x = np.arange(npix)
        omodel._s2 = interp1d(x,model._s2,kind='cubic',bounds_error=False,
                           fill_value=(np.nan,np.nan),assume_sorted=True)(xout)
        omodel._scales = model._scales
        omodel._theta = np.zeros((npix2,npars),np.float64)
        for i in range(npars):
            omodel._theta[:,i] = interp1d(x,model._theta[:,i],kind='cubic',bounds_error=False,
                           fill_value=(np.nan,np.nan),assume_sorted=True)(xout)
        omodel._design_matrix = model._design_matrix
        omodel._fiducials = model._fiducials
        if model.dispersion is not None:
            omodel.dispersion = interp1d(x,model.dispersion,kind='cubic',bounds_error=False,
                           fill_value=(np.nan,np.nan),assume_sorted=True)(xout)
        omodel.regularization = model.regularization
        if hasattr(model,'ranges') is True: omodel.ranges=model.ranges
        
    return omodel


# Convolve a CannonModel model
def convolve_cannon_model(model,lsf):
    """
    Convolve a Cannon model or list of models with an input LSF.

    Parameters
    ----------
    model : Cannon model or list
      Input Cannon model or list of Cannon models to convolve, with Npix pixels.
    lsf : array
      2D line spread function (LSF) to convolve with the Cannon model.
      The shape must be [Npix,Nlsf], where Npix is the same as the number of
      pixels in the Cannon model.

    Return
    ------
    omodel : Cannon model or list
      The convolved Cannon model or list of models.

    Examples
    --------

    omodel = convolve_cannon_model(model,lsf)

    """
    
    #  Need to allow this to be vary with wavelength
    # lsf can be a 1D kernel, or a 2D LSF array that gives
    #  the kernel for each pixel separately
    
    if type(model) is list:
        omodel = []
        for i in range(len(model)):
            model1 = model[i]
            omodel1 = convolve_cannon_model(model1,lsf)
            omodel.append(omodel1)
    else:
        npix, npars = model.theta.shape
        nlabels = len(model.vectorizer.label_names)
        labelled_set = np.zeros([2,nlabels])
        normalized_flux = np.zeros([2,npix])
        normalized_ivar = normalized_flux.copy()*0
        omodel = tc.CannonModel(labelled_set,normalized_flux,normalized_ivar,model.vectorizer)
        omodel._theta = model._theta*0
        #omodel._theta = np.zeros((npix2,npars),np.float64)
        if lsf.ndim==1:
            omodel._s2 = convolve(model._s2,lsf,mode="reflect")
            for i in range(npars):
                omodel._theta[:,i] = convolve(model._theta[:,i],lsf,mode="reflect")
        else:
            omodel._s2 = utils.convolve_sparse(model._s2,lsf)
            for i in range(npars):
                omodel._theta[:,i] = utils.convolve_sparse(model._theta[:,i],lsf)
        omodel._scales = model._scales
        omodel._design_matrix = model._design_matrix
        omodel._fiducials = model._fiducials
        if model.dispersion is not None:
            omodel.dispersion = model.dispersion
        omodel.regularization = model.regularization
        if hasattr(model,'ranges') is True: omodel.ranges=model.ranges
        
    return omodel


# Prepare the cannon model for a specific observed spectrum
def prepare_cannon_model(model,spec,dointerp=False):
    """
    This prepares a Cannon model or list of models for an observed spectrum.
    The models are trimmed, rebinned, convolved to the spectrum LSF's and
    finally interpolated onto the spectrum's wavelength scale.

    The final interpolation can be omitted by setting dointerp=False (the default).
    This can be useful if the model is to interpolated multiple times on differen
    wavelength scales (e.g. for multiple velocities or logarithmic wavelength
    scales for cross-correlation).

    Parameters
    ----------
    model : Cannon model or list
       The Cannon model(s) to prepare for "spec".
    spec : Spec1D object
       The observed spectrum for which to prepare the Cannon model(s).
    dointerp : bool, optional
       Do the interpolation onto the observed wavelength scale.
       The default is Fal.se

    Returns
    -------
    omodel : Cannon model or list
        The Cannon model or list of models prepared for "spec".

    Examples
    --------

    omodel = prepare_cannon_model(model,spec)

    """
    
    if spec.wave is None:
        raise Exception('No wavelength in observed spectrum')
    if spec.lsf is None:
        raise Exception('No LSF in observed spectrum')
    if type(model) is not list:
        if model.dispersion is None:
            raise Exception('No model wavelength information')

    
    if type(model) is list:
        outmodel = []
        for i in range(len(model)):
            model1 = model[i]
            omodel1 = prepare_cannon_model(model1,spec,dointerp=dointerp)
            outmodel.append(omodel1)
    else:
        # Observed spectrum values
        wave = spec.wave
        ndim = wave.ndim
        if ndim==2:
            npix,norder = wave.shape
        if ndim==1:
            norder = 1
            npix = len(wave)
            wave = wave.reshape(npix,norder)
        # Loop over the orders
        outmodel = []
        for o in range(norder):
            w = wave[:,o]
            w0 = np.min(w)
            w1 = np.max(w)
            dw = dln.slope(w)
            dw = np.hstack((dw,dw[-1]))
            dw = np.abs(dw)
            meddw = np.median(dw)
            npix = len(w)

            if (np.min(model.dispersion)>w0) | (np.max(model.dispersion)<w1):
                raise Exception('Model does not cover the observed wavelength range')
            
            # Trim
            if (np.min(model.dispersion)<(w0-50*meddw)) | (np.max(model.dispersion)>(w1+50*meddw)):
                tmodel = trim_cannon_model(model,w0=w0-20*meddw,w1=w1+20*meddw)
            #if (np.min(model.dispersion)<(w0-1000.0*w0/cspeed)) | (np.max(model.dispersion)>(w1+1000.0*w1/cspeed)):                                
            #    # allow for up to 1000 km/s offset on each end
            #    tmodel = trim_cannon_model(model,w0=w0-1000.0*w0/cspeed,w1=w1+1000.0*w1/cspeed)
                tmodel.trim = True
            else:
                tmodel = cannon_copy(model)
                tmodel.trim = False


            print('KLUDGE')
            return tmodel
                
            # Rebin
            #  get LSF FWHM (A) for a handful of positions across the spectrum
            xp = np.arange(npix//20)*20
            fwhm = spec.lsf.fwhm(w[xp],xtype='Wave',order=o)
            # FWHM is in units of lsf.xtype, convert to wavelength/angstroms, if necessary
            if spec.lsf.xtype.lower().find('pix')>-1:
                dw1 = dln.interp(w,dw,w[xp],assume_sorted=False)
                fwhm *= dw1
            #  convert FWHM (A) in number of model pixels at those positions
            dwmod = dln.slope(tmodel.dispersion)
            dwmod = np.hstack((dwmod,dwmod[-1]))
            xpmod = interp1d(tmodel.dispersion,np.arange(len(tmodel.dispersion)),kind='cubic',bounds_error=False,
                             fill_value=(np.nan,np.nan),assume_sorted=False)(w[xp])
            xpmod = np.round(xpmod).astype(int)
            fwhmpix = fwhm/dwmod[xpmod]
            # need at least ~3 pixels per LSF FWHM across the spectrum
            nbin = np.round(np.min(fwhmpix)//3).astype(int)
            if nbin==0:
                raise Exception('Model has lower resolution than the observed spectrum')
            if nbin>1:
                rmodel = rebin_cannon_model(tmodel,nbin)
                rmodel.rebin = True
            else:
                rmodel = cannon_copy(tmodel)
                rmodel.rebin = False
                
            # Convolve
            lsf = spec.lsf.anyarray(rmodel.dispersion,xtype='Wave',order=0)

            import pdb; pdb.set_trace()
            cmodel = convolve_cannon_model(rmodel,lsf)
            cmodel.convolve = True
        
            # Interpolate
            if dointerp is True:
                omodel = interp_cannon_model(cmodel,wout=spec.wave)
                omodel.interp = True
            else:
                omodel = cannon_copy(cmodel)
                omodel.interp = False

            # Order information
            omodel.norder = norder
            omodel.order = o
                
            # Append to final output
            outmodel.append(omodel)

    # Single-element list
    if (type(outmodel) is list) & (len(outmodel)==1):
        outmodel = outmodel[0] 
            
    return outmodel


def model_spectrum(models,spec,teff=None,logg=None,feh=None,rv=None):
    """
    Create a Cannon model spectrum matches to an input observed spectrum
    and for given stellar parameters (teff, logg, feh) and radial velocity
    (rv).

    Parameters
    ----------
    models : Cannon model or list
       The Cannon models or list of models to use.
    spec : Spec1D object
       The observed spectrum for which to create the model spectrum.
    teff : float
       The desired effective temperature.
    logg : float
       The desired surface gravity.
    feh : float
       The desired metallicity, [Fe/H].
    rv : float
       The desired radial velocity (km/s).

    Returns
    -------
    mspec : Spec1D object
      The final Cannon model spectrum (as Spec1D object) with the
      specified stellar parameters and RV prepared for the input "spec"
      observed spectrum.

    Examples
    --------

    mspec = model_spectrum(models,spec,teff=4500.0,logg=4.0,feh=-1.2,rv=55.0)

    """
    
    if teff is None:
        raise Exception("Need to input TEFF")    
    if logg is None:
        raise Exception("Need to input LOGG")
    if feh is None:
        raise Exception("Need to input FEH")
    if rv is None: rv=0.0
    
    pars = np.array([teff,logg,feh])
    pars = pars.flatten()
    # Get best cannon model
    model = get_best_cannon_model(models,pars)
    if model is None: return None
    # Create the model spectrum

    # Loop over the orders
    model_spec = np.zeros(spec.wave.shape,dtype=np.float32)
    model_wave = np.zeros(spec.wave.shape,dtype=np.float64)
    sigma = np.zeros(spec.wave.shape,dtype=np.float32)
    for o in range(spec.norder):
        if spec.ndim==1:
            model1 = model
            spwave1 = spec.wave
            sigma1 = spec.lsf.sigma(xtype='Wave')
        else:
            model1 = model[o]
            spwave1 = spec.wave[:,o]
            sigma1 = spec.lsf.sigma(xtype='Wave',order=o)            
        model_spec1 = model1(pars)
        model_wave1 = model1.dispersion.copy()

        # Apply doppler shift to wavelength
        if rv!=0.0:
            model_wave1 *= (1+rv/cspeed)
            # w = synwave*(1.0d0 + par[0]/cspeed)
    
        # Interpolation
        model_spec_interp = interp1d(model_wave1,model_spec1,kind='cubic',bounds_error=False,
                                     fill_value=(np.nan,np.nan),assume_sorted=True)(spwave1)

        # UnivariateSpline
        #   this is 10x faster then the interpolation
        # scales linearly with number of points
        #spl = UnivariateSpline(model_wave,model_spec)
        #model_spec_interp = spl(spec.wave)
        
        # Stuff in the information for this order
        model_spec[:,o] = model_spec_interp
        model_wave[:,o] = spwave1
        sigma[:,o] = sigma1
        
    # Create Spec1D object
    mspec = Spec1D(model_spec,wave=model_wave,lsfsigma=sigma,instrument='Model')
    mspec.teff = teff
    mspec.logg = logg
    mspec.feh = feh
    mspec.rv = rv
    mspec.snr = np.inf
        
    return mspec

