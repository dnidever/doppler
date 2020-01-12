#!/usr/bin/env python

"""CANNON.PY - Routines to work with Cannon models.

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20190922'  # yyyymmdd

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
    

# Load the cannon model
def load_all_cannon_models():
    fil = os.path.abspath(__file__)
    codedir = os.path.dirname(fil)
    datadir = codedir+'/data/'
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
        models.append(model1)
    return models

# Get correct cannon model for a given set of inputs
def get_best_cannon_model(models,pars):
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

# Convolve a CannonModel model with a wavelength-dependent Gaussian
def gconvolve_cannon_model(model,xgcoef=None,wgcoef=None):
    # gcoef - array of coefficients that returns Gaussian sigma (pixels)
    #           input is pixels.  values in ascending order
    #           e.g. gcoef = [c0, c1, c2, ...]
    # wgcoef - array of coefficients that returns Gaussian sigma (A)
    #           input is wavelength (A).  values in ascending order
    #           e.g. gcoef = [c0, c1, c2, ...]    
    
    #  Need to allow this to be vary with wavelength

    if (xgcoef is None) & (wgcoef is None):
        raise Exception('xgcoef or wgcoef must be input')
    
    if type(model) is list:
        omodel = []
        for i in range(len(model)):
            model1 = model[i]
            omodel1 = gconvolve_cannon_model(model1,xgcoef=xgcoef,wgcoef=wgoef)
            omodel.append(omodel1)
    else:

        if (wgcoef is not None) & (model.dispersion is None):
            raise Exception('wgcoef input but no dispersion information in model')

        npix, npars = model.theta.shape
        x = np.arange(npix)
        
        # Calculate xsigma from wgcoef
        if (xgcoef is None) & (wgcoef is not None):
            dw = dln.slope(model.dispersion)
            dw = np.hstack((dw,dw[-1]))
            wsigma = np.polyval(wgcoef[::-1],model.dispersion)
            xsigma = wsigma / dw

        # Calculate xsigma from xgcoef
        if (wgcoef is None) & (xgcoef is not None):
            xsigma = np.polyval(xgcoef[::-1],x)

        # Figure out nLSF pixels needed, +/-3 sigma
        nlsf = np.int(np.round(np.max(xsigma)*6))
        if nlsf % 2 == 0: nlsf+=1                   # must be even
        
        # Make LSF array
        lsf = np.zeros((npix,nlsf))
        xlsf = np.arange(nlsf)-nlsf//2
        #for i in range(npix):
        #    lsf[i,:] = np.exp(-0.5*xlsf**2 / xsigma[i]**2)
        #    lsf[i,:] /= np.sum(lsf[i,:])             # normalize
        xlsf2 = np.repeat(xlsf,npix).reshape((nlsf,npix)).T
        xsigma2 = np.repeat(xsigma,nlsf).reshape((npix,nlsf))
        lsf = np.exp(-0.5*xlsf2**2 / xsigma2**2) / (np.sqrt(2*np.pi)*xsigma2)

        nlabels = len(model.vectorizer.label_names)
        labelled_set = np.zeros([2,nlabels])
        normalized_flux = np.zeros([2,npix])
        normalized_ivar = normalized_flux.copy()*0
        omodel = tc.CannonModel(labelled_set,normalized_flux,normalized_ivar,model.vectorizer)
        omodel._s2 = utils.convolve_sparse(model._s2,lsf)
        omodel._scales = model._scales
        omodel._theta = np.zeros((npix,npars),np.float64)
        for i in range(npars):
            omodel._theta[:,i] = utils.convolve_sparse(model._theta[:,i],lsf)
        omodel._design_matrix = model._design_matrix
        omodel._fiducials = model._fiducials
        if model.dispersion is not None:
            omodel.dispersion = model.dispersion
        omodel.regularization = model.regularization
        if hasattr(model,'ranges') is True: omodel.ranges=model.ranges
        
    return omodel


# Prepare the cannon model for a specific observed spectrum
def prepare_cannon_model(model,spec,dointerp=False):

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
            npix = len(w)

            if (np.min(model.dispersion)>w0) | (np.max(model.dispersion)<w1):
                raise Exception('Model does not cover the observed wavelength range')

            # Trim
            if (np.min(model.dispersion)<(w0-50*dw[0])) | (np.max(model.dispersion)>(w1+50*dw[-1])):
                tmodel = trim_cannon_model(model,w0=w0-20*dw[0],w1=w1+20*dw[-1])
                tmodel.trim = True
            else:
                tmodel = cannon_copy(model)
                tmodel.trim = False

            # Rebin
            #  get LSF FWHM (A) for a handful of positions across the spectrum
            xp = np.arange(npix//20)*20
            fwhm = spec.lsf.fwhm(w[xp],xtype='Wave',order=o)
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
            model_wave *= (1+rv/cspeed)
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
        model_spec[...] = model_spec_interp
        model_wave[...] = spwave1
        sigma[...] = sigma1
        
    # Create Spec1D object
    mspec = Spec1D(model_spec,wave=model_wave,lsfsigma=sigma,instrument='Model')
    mspec.teff = teff
    mspec.logg = logg
    mspec.feh = feh
    mspec.rv = rv
    mspec.snr = np.inf
        
    return mspec

def model_spectrum_rvfit(models,spec,teff=None,logg=None,feh=None):
    if teff is None:
        raise Exception("Need to input TEFF")    
    if logg is None:
        raise Exception("Need to input LOGG")
    if feh is None:
        raise Exception("Need to input FEH")
    
    pars = np.array([teff,logg,feh])
    # Get best cannon model
    model = get_best_cannon_model(models,pars)
    if model is None:
        print('Outside the parameter range')
        return
    # Create the model spectrum
    model_spec = model(pars)
    model_wave = model.dispersion.copy()
    npix = len(model_spec)

    # Sample +/-800 km/s in 10 km/s steps
    #  interpolate all of the wavelengths at once to increase speed
    #model_wave *= (1+rv/cspeed)

    nrv = 161
    rvstep = 10.0
    rv0 = -800.0
    nopix = len(spec.wave)
    wave1 = spec.wave.copy()
    wave2 = np.repeat(wave1,nrv).reshape((nopix,nrv)).T
    rv = np.arange(nrv)*rvstep+rv0
    dop = (1+rv/cspeed)
    dop2 = np.repeat(dop,nopix).reshape((nrv,nopix))
    dopwave = wave2*dop2

    # Interpolation
    model_spec_interp = interp1d(model_wave,model_spec,kind='cubic',bounds_error=False,
                                 fill_value=(np.nan,np.nan),assume_sorted=True)(dopwave)

    # Calculate chisq
    flux = np.repeat(spec.flux.copy(),nrv).reshape((nopix,nrv)).T
    err = np.repeat(spec.err.copy(),nrv).reshape((nopix,nrv)).T    

    chisq = np.sum((flux-model_spec_interp)**2/err**2,axis=1)
    bestind = np.argmin(chisq)
    bestrv = rv[bestind]
    bestchisq = chisq[bestind]

    # Interpolate to better RV
    rv2 = np.arange(201)*0.1-10 + bestrv
    chisq2 = dln.interp(rv,chisq,rv2)
    bestind2 = np.argmin(chisq2)
    bestrv2 = rv2[bestind2]
    bestchisq2 = chisq2[bestind2]
    
    print('RV = '+str(bestrv2)+' km/s')
    #plt.plot(rv,chisq)

    best_model = model_spec_interp[bestind,:]
    #best_model = model_spectrum(models,spec,teff=None,logg=None,feh=None,rv=None):    
    plt.plot(spec.flux)
    plt.plot(best_model)
    
    return (bestrv,bestchisq)

    # SPECTRUM MUST BE NORMALIZED!!!

    # UnivariateSpline
    #   this is 10x faster then the interpolation
    # scales linearly with number of points
    #spl = UnivariateSpline(model_wave,model_spec)
    #model_spec_interp = spl(spec.wave)


    
