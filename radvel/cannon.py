#!/usr/bin/env python

"""RV.PY - Generic Radial Velocity Software

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
from scipy.ndimage.filters import median_filter,gaussian_filter1d,convolve
from scipy.optimize import curve_fit, least_squares
from scipy.special import erf
from scipy.interpolate import interp1d
from scipy import sparse
#from numpy.polynomial import polynomial as poly
#from lmfit import Model
#from apogee.utils import yanny, apload
#from sdss_access.path import path
import thecannon as tc
#import bindata
#from utils import *
from dlnpyutils import utils as dln, bindata

# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# astropy.modeling can handle errors and constraints

def sparsify(lsf):
    # sparsify
    # make a sparse matrix
    # from J.Bovy's lsf.py APOGEE code
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
    # convolution with matrices
    # from J.Bovy's lsf.py APOGEE code    
    # spec - [npix]
    # lsf - [npix,nlsf]
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
        
# Load the cannon model
def load_all_cannon_models():
    fil = os.path.abspath(__file__)
    codedir = os.path.dirname(fil)
    datadir = os.path.dirname(codedir)+'/data/'
    files = glob(datadir+'cannongrid*.pkl')
    nfiles = len(files)
    if nfiles==0:
        raise Exception("No Cannon model files in "+datadir)
    models = []
    for f in files:
        model1 = tc.CannonModel.read(f)
        ranges = np.zeros([3,2])
        for i in range(3):
            ranges[i,:] = minmax(model1._training_set_labels[:,i])
        model1.ranges = ranges
        models.append(model1)
    return models

# Get correct cannon model for a given set of inputs
def get_best_cannon_model(models,pars):
    n = len(models)
    for m in models:
        #inside.append(m.in_convex_hull(pars))        
        # in_convex_hull() is very slow
        inside = True
        for i in range(3):
            inside &= (pars[i]>=m.ranges[i,0]) & (pars[i]<=m.ranges[i,1])
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
        npix2 = npix // binsize
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

    return omodel

# Convolve a CannonModel model
def convolve_cannon_model(model,kernel):

    #  Need to allow this to be vary with wavelength
    
    if type(model) is list:
        omodel = []
        for i in range(len(model)):
            model1 = model[i]
            omodel1 = convolve_cannon_model(model1,kernel)
            omodel.append(omodel1)
    else:
        npix, npars = model.theta.shape
        nlabels = len(model.vectorizer.label_names)
        labelled_set = np.zeros([2,nlabels])
        normalized_flux = np.zeros([2,npix])
        normalized_ivar = normalized_flux.copy()*0
        omodel = tc.CannonModel(labelled_set,normalized_flux,normalized_ivar,model.vectorizer)
        omodel._s2 = convolve(model._s2,kernel,mode="reflect")
        omodel._scales = model._scales
        omodel._theta = np.zeros((npix,npars),np.float64)
        for i in range(npars):
            omodel._theta[:,i] = convolve(model._theta[:,i],kernel,mode="reflect")
        omodel._design_matrix = model._design_matrix
        omodel._fiducials = model._fiducials
        if model.dispersion is not None:
            omodel.dispersion = model.dispersion
        omodel.regularization = model.regularization

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
        omodel._s2 = convolve_sparse(model._s2,lsf)
        omodel._scales = model._scales
        omodel._theta = np.zeros((npix,npars),np.float64)
        for i in range(npars):
            omodel._theta[:,i] = convolve_sparse(model._theta[:,i],lsf)
        omodel._design_matrix = model._design_matrix
        omodel._fiducials = model._fiducials
        if model.dispersion is not None:
            omodel.dispersion = model.dispersion
        omodel.regularization = model.regularization

    return omodel


def model_spectrum(models,teff=None,logg=None,feh=None,wave=None,w0=None,w1=None,dw=None,fwhm=None,trim=False):
    if teff is None:
        raise Exception("Need to input TEFF")    
    if logg is None:
        raise Exception("Need to input LOGG")
    if feh is None:
        raise Exception("Need to input FEH")    
    
    pars = np.array([teff,logg,feh])
    # Get best cannon model
    model = get_best_cannon_model(models,pars)

    npix = model.dispersion.shape[0]
    dw0 = model.dispersion[1]-model.dispersion[0]

    # Wave array input
    if wave is not None:
        w0 = np.min(wave)
        w1 = np.max(wave)
        dw = np.min(slope(wave))
    
    # Defaults for w0, w1, dw
    if w0 is None:
        w0 = np.min(model.dispersion)
    if w1 is None:
        w1 = np.max(model.dispersion)
    if dw is None:
        dw = dw0

    # Trim the model
    if trim is True:
        # Get trimmed Cannon model
        model2 = trim_cannon_model(model,w0=w0-10*dw,w1=w1+10*dw)
        npix2 = model2.dispersion.shape[0]
        # Get the spectrum
        wave1 = model2.dispersion
        flux1 = model2(pars)
    else:
        wave1 = model.dispersion
        flux1 = model(pars)
        npix2 = npix
        
    # Rebin
    if dw != dw0:
        binsize = int(dw // dw0)
        newnpix = int(npix2 // binsize)
        # Need to fix this to include w0 and w1 if possible
        wave2 = rebin(wave1[0:newnpix*binsize],newnpix)
        flux2 = rebin(flux1[0:newnpix*binsize],newnpix)
    else:
        wave2 = wave1
        flux2 = flux1

    # Smoothing
    if fwhm is not None:
        fwhmpix = fwhm/dw
        flux2 = gsmooth(flux2,fwhmpix)
    # Interpolation
    if wave is not None:
        tflux2 = flux2.copy()
        f = interp1d(wave2,tflux2,kind='cubic',bounds_error=False,fill_value=(0.0,0.0),assume_sorted=True)
        flux2 = f(wave)
        del(tflux2)
        wave2 = wave.copy()

    # Create Spec1D object
    spec = Spec1D(flux2,wave=wave2,instrument='Model')
    spec.teff = teff
    spec.logg = logg
    spec.feh = feh
    if fwhm is not None:
        spec.fwhm = fwhm
    spec.snr = np.inf
        
    return spec
  
