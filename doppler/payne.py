#!/usr/bin/env python

"""PAYNE.PY - Routines to work with Payne models.

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20210605'  # yyyymmdd

# Some of the software is from Yuan-Sen Ting's The_Payne repository
# https://github.com/tingyuansen/The_Payne

import os
import numpy as np
import warnings
from glob import glob
from scipy.interpolate import interp1d
from dlnpyutils import (utils as dln, bindata, astro)
from .spec1d import Spec1D
from . import utils
import copy
import logging
import contextlib, io, sys
import time
try:
    import __builtin__ as builtins # Python 2
except ImportError:
    import builtins # Python 3
    
# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

cspeed = 2.99792458e5  # speed of light in km/s

# Get print function to be used locally, allows for easy logging
print = utils.getprintfunc() 

    
def leaky_relu(z):
    '''
    This is the activation function used by default in all our neural networks.
    '''

    return z*(z > 0) + 0.01*z*(z < 0)
    


# Load the default Payne model
def load_model():
    """
    Load the default Payne model.
    Return as PayneModel or PayneModelSet.
    """

    datadir = utils.datadir()
    files = glob(datadir+'payne_coolhot_*.npz')
    nfiles = len(files)
    if nfiles==0:
        # No Payne model files.  Download them
        print("No Payne model files in "+datadir+". Downloading them.")
        utils.download_data()
        #raise Exception("No Payne model files in "+datadir)
    if nfiles>1:
        return PayneModelSet.read(files)
    else:
        return PayneModel.read(files)


# Load a single or list of Payne models
def load_payne_model(mfile):
    """
    Load a Payne model from file.

    Returns
    -------
    mfiles : string
       File name (or list of filenames) of Payne models to load.

    Examples
    --------
    model = load_payne_model()

    """

    if os.path.exists(mfile) == False:
        raise ValueError(mfile+' not found')

    
    # read in the weights and biases parameterizing a particular neural network. 
    tmp = np.load(mfile)
    w_array_0 = tmp["w_array_0"]
    w_array_1 = tmp["w_array_1"]
    w_array_2 = tmp["w_array_2"]
    b_array_0 = tmp["b_array_0"]
    b_array_1 = tmp["b_array_1"]
    b_array_2 = tmp["b_array_2"]
    x_min = tmp["x_min"]
    x_max = tmp["x_max"]
    if 'wavelength' in tmp.files:
        wavelength = tmp["wavelength"]
    else:
        print('WARNING: No wavelength array')
        wavelength = np.arange(w_array_2.shape[0]).astype(np.float64)  # dummy wavelengths        
    if 'labels' in tmp.files:
        labels = list(tmp["labels"])
    else:
        print('WARNING: No label array')
        labels = [None] * w_array_0.shape[1]
    if 'wavevac' in tmp.files:
        wavevac = bool(tmp["wavevac"])
    else:
        print('WARNING: No wavevac')
        wavevac = False
    coeffs = (w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max)
    tmp.close()
    return coeffs, wavelength, labels, wavevac

def load_models():
    """
    Load all Payne models from the Doppler data/ directory
    and return as a DopplerPayneModel.
    Download the data files, if needed.

    Returns
    -------
    models : DopplerPayneModel
        DopplerPayneModel for all Payne models in the
        Doppler /data directory.

    Examples
    --------
    models = load_models()

    """    
    datadir = utils.datadir()
    files = glob(datadir+'payne_coolhot_*.npz')
    nfiles = len(files)
    if nfiles==0:
        # No Payne model files.  Download them
        print("No Payne model files in "+datadir+". Downloading them.")
        utils.download_data()
        #raise Exception("No Payne model files in "+datadir)
    return DopplerPayneModel.read(files)

def check_params(model,params):
    """ Check input fit or fixed parameters against Payne model labels."""
    # Check the input labels against the Paybe model labels

    if isinstance(params,dict):
        paramdict = params.copy()
        params = list(paramdict.keys())
        isdict = True
    else:
        isdict = False

    # Check for duplicates
    uparams = np.unique(np.array(params))
    if len(uparams)!=len(params):
        raise ValueError('There are duplicates in '+','.join(params))
        
    # Loop over parameters
    for i,par in enumerate(params):
        # replace VROT with VSINI        
        if par=='VROT' and 'VSINI' in model.labels:
            print('Replacing VROT -> VSINI')
            params[i] = 'VSINI'
            par = 'VSINI'
        # replace VMICRO with VTURB            
        elif par=='VMICRO' and 'VTURB' in model.labels:
            print('Replacing VMICRO -> VTURB')
            params[i] = 'VTURB'
            par = 'VTURB'
        # check against model labels
        if (par != 'ALPHA_H') and (not par in model.labels):
            raise ValueError(par+' NOT a Payne label. Available labels are '+','.join(model.labels)+' and ALPHA_H')

    # Return "adjusted" params
    if isdict==True:
        paramdict = dict(zip(params,paramdict.values()))
        return paramdict
    else:    
        return params


def prepare_payne_model(model,labels,spec,rv=None,vmacro=None,vsini=None,wave=None,lsfout=False):
    """
    Prepare a Payne spectrum for a given observed spectrum.

    Parameters
    ----------
    model : Payne model
        Payne model to use.
    labels : list
        List of Payne parameter/label names to fit.
    spec : Spec1D object
        The observed spectrum.
    rv : float, optional
        Doppler shift to apply to the Payne model (in km/s).  Default is 0.0.
    vmacro : float, optional
        Extra Gaussian broadening to apply to Payne model (in km/s) for macroturbulence.
        Default is 0.0.
    vsini : float, optional
        Rotational broadening to apply to the Payne model (in km/s).  Default is 0.0.
    wave : numpy array, optional
        Input wavelength array to use for the output Payne model.  Default is to use the
        observed spectrum wavelengths.
    lsfout : boolean, optional
        Set to output the 2D LSF and wavelength array for the Payne model.

    Returns
    -------
    outmodel : Spec1D object
        The output Payne model spectrum convolved to the LSF of the observd spectrum.
    lsf_list : list
        List of output 2D LSF array, one element per spectral order.  Only if lsfout=True.
    lsfwave_list : list
        List of output 2D LSF wavelength array, one element per spectral order.  Only if lsfout=True.

    Example
    -------

    .. code-block:: python

         outmodel = prepare_payne_model(model,labels,spec)

    """
    
    # Convert wavelength from air->vacuum or vice versa
    if model.wavevac != spec.wavevac:
        model.wavevac = spec.wavevac   # this will set things behind the scenes

    # Get full wavelength range and total wavelength coverage in the orders
    owr = spec.wrange
    owavefull = dln.valrange(owr)
    owavechunks = 0.0
    odw = np.zeros(spec.norder,np.float64)
    specwave = spec.wave.copy()
    specmask = spec.mask.copy()    
    if spec.wave.ndim==1:  # make 2D with order in second dimension
        specwave = np.atleast_2d(specwave).T
        specmask = np.atleast_2d(specmask).T
    for o,sp in enumerate(spec):
        owavechunks += dln.valrange(sp.wrange)
        odw[o] = np.median(dln.slope(sp.wave))
        
    # Check input WAVE array
    if wave is not None:
        if wave.ndim==1:
            wnorder = 1
            wave = np.atleast_2d(wave).T
        else:
            wnorder = wave.shape[1]
        if wnorder != spec.norder:
            raise ValueError('Wave must have same orders as Spectrum')
        
    # Get model spectrum for the entire wavelength range, across all orders
    # Or do them separately, if there are large gaps (e.g GALAH)
    if owavechunks > 0.5*owavefull:
        model_all_in_one = True
        wextend = 0.0
        for o,sp in enumerate(spec):
            dw = dln.slope(sp.wave)
            dw = np.hstack((dw,dw[-1]))
            dw = np.abs(dw)
            meddw = np.median(dw)
            nextend = int(np.ceil(len(dw)*0.25))  # extend 25% on each end
            nextend = np.maximum(nextend,200)     # or 200 pixels
            wextend = np.max([wextend,nextend*meddw])
        w0 = np.maximum( owr[0]-wextend, np.min(model.dispersion))
        w1 = np.minimum( owr[1]+wextend, np.max(model.dispersion))
        modelspec_all = model(labels,wr=[w0,w1])
    else:
        model_all_in_one = False

    # Loop over the orders
    outmodel = spec.copy()
    outmodel.err[:] = 0
    outmodel.mask[:] = False
    if wave is not None:
        outmodel.flux = np.zeros(wave.shape,float)
        outmodel.err = np.zeros(wave.shape,float)
        outmodel.mask = np.ones(wave.shape,bool)        
    lsf_list = []
    lsfwave_list = []
    for o,sp in enumerate(spec):
        w = sp.wave
        m = sp.mask
        gdpix, = np.where(~m)
        if len(gdpix)==0:
            raise ValueError('No unmasked pixels in order ',o)
        w = w[gdpix]
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
        nextend = int(np.ceil(len(w)*0.25))  # extend 25% on each end
        nextend = np.maximum(nextend,200)    # or 200 pixels
        if model_all_in_one == True:
            if (np.min(model.dispersion)<(w0-nextend*meddw)) | (np.max(model.dispersion)>(w1+nextend*meddw)):            
                gd, = np.where( (modelspec_all.wave >= (w0-nextend*meddw)) &
                                (modelspec_all.wave <= (w1+nextend*meddw)) )
                ngd = len(gd)
                tmodelflux = modelspec_all.flux[gd]
                tmodelwave = modelspec_all.wave[gd]
            else:
                tmodelflux = modelspec_all.flux
                tmodelwave = modelspec_all.wave                
        else:
            modelspec = model(labels,wr=[w0-nextend*meddw,w1+nextend*meddw])
            tmodelflux = modelspec.flux
            tmodelwave = modelspec.wave
            
        # Rebin
        #  get LSF FWHM (A) for a handful of positions across the spectrum
        xp = np.arange(npix//20)*20
        fwhm = spec.lsf.fwhm(w[xp],xtype='Wave',order=o)
        # FWHM is in units of lsf.xtype, convert to wavelength/angstroms, if necessary
        if spec.lsf.xtype.lower().find('pix')>-1:
            dw1 = dln.interp(w,dw,w[xp],assume_sorted=False)
            fwhm *= dw1
        #  convert FWHM (A) in number of model pixels at those positions
        dwmod = dln.slope(tmodelwave)
        dwmod = np.hstack((dwmod,dwmod[-1]))
        xpmod = interp1d(tmodelwave,np.arange(len(tmodelwave)),kind='cubic',bounds_error=False,
                         fill_value=(np.nan,np.nan),assume_sorted=False)(w[xp])
        xpmod = np.round(xpmod).astype(int)
        fwhmpix = fwhm/dwmod[xpmod]
        # need at least ~4 pixels per LSF FWHM across the spectrum
        #  using 3 affects the final profile shape
        nbin = np.round(np.min(fwhmpix)//4).astype(int)
        if np.min(fwhmpix) < 3.7:
            warnings.warn('Model has lower resolution than the observed spectrum. Only '+str(np.min(fwhmpix))+' model pixels per resolution element') 
        if np.min(fwhmpix) < 2.8:
            raise Exception('Model has lower resolution than the observed spectrum. Only '+str(np.min(fwhmpix))+' model pixels per resolution element')

        if nbin>1:
            npix2 = np.round(len(tmodelflux) // nbin).astype(int)
            rmodelflux = dln.rebin(tmodelflux[0:npix2*nbin],npix2)
            rmodelwave = dln.rebin(tmodelwave[0:npix2*nbin],npix2)
        else:
            rmodelflux = tmodelflux
            rmodelwave = tmodelwave
                
        # Convolve with LSF, Vsini and Vmacro kernels
        if model._lsf is not None:
            lsf = model._lsf[o]
            if lsf.shape[0] != len(rmodelwave):
                print('prepare_payne_model: saved LSF does not match')
                import pdb; pdb.set_trace()
        else:
            lsf = spec.lsf.anyarray(rmodelwave,xtype='Wave',order=o,original=False)
        lsf_list.append(lsf)
        lsfwave_list.append(rmodelwave)
        cmodelflux = utils.convolve_sparse(rmodelflux,lsf)
        # Apply Vmacro broadening and Vsini broadening (km/s)
        if (vmacro is not None) | (vsini is not None):
            cmodelflux = utils.broaden(rmodelwave,cmodelflux,vgauss=vmacro,vsini=vsini)
            
        # Apply Radial Velocity
        if rv is not None:
            if rv != 0.0: rmodelwave *= (1+rv/cspeed)
        
        # Interpolate
        if wave is None:
            omodelflux = interp1d(rmodelwave,cmodelflux,kind='cubic',bounds_error=False,
                                  fill_value=(np.nan,np.nan),assume_sorted=True)(w)
        else:
            omodelflux = interp1d(rmodelwave,cmodelflux,kind='cubic',bounds_error=False,
                                  fill_value=(np.nan,np.nan),assume_sorted=True)(wave[:,o])

        if outmodel.flux.ndim==1:
            outmodel.flux[0:len(omodelflux)] = omodelflux
        else:
            outmodel.flux[0:len(omodelflux),o] = omodelflux        
            
    if wave is not None:
        outmodel.wave = wave.copy()

    # Change single order 2D arrays to 1D
    if spec.norder==1:
        outmodel.flux = outmodel.flux.flatten()
        outmodel.err = outmodel.err.flatten()        
        outmodel.wave = outmodel.wave.flatten()
        outmodel.mask = outmodel.mask.flatten()        
        
    if lsfout is True:
        return outmodel,lsf_list,lsfwave_list
    else:
        return outmodel

    
def mkdxlim(fitparams):
    """
    Make array of parameter changes at which curve_fit should finish.

    Parameters
    ----------
    fitparams : list
        List of parameter names.

    Returns
    -------
    dx_lim : numpy array
       Array of parameter changes at which curve_fit should finish.

    Example
    -------
    .. code-block:: python

         dx_lim = mkdxlim(fitparams)

    """
    npar = len(fitparams)
    dx_lim = np.zeros(npar,float)
    for k in range(npar):
        if fitparams[k]=='TEFF':
            dx_lim[k] = 1.0
        elif fitparams[k]=='LOGG':
            dx_lim[k] = 0.005
        elif (fitparams[k]=='VMICRO' or fitparams[k]=='VTURB'):
            dx_lim[k] = 0.1
        elif (fitparams[k]=='VSINI' or fitparams[k]=='VROT'):
            dx_lim[k] = 0.1
        elif fitparams[k]=='VMACRO':
            dx_lim[k] = 0.1
        elif fitparams[k]=='RV':
            dx_lim[k] = 0.01
        elif fitparams[k].endswith('_H'):
            dx_lim[k] = 0.005
        else:
            dx_lim[k] = 0.01
    return dx_lim

def mkinitlabels(labels):
    """
    Make initial guesses for Payne labels.

    Parameters
    ----------
    labels : list
        List of parameter names.

    Returns
    -------
    initpars : numpy array
       Array of initial label values.

    Example
    -------
    .. code-block:: python

         initpars = mkinitlabels(labels)

    """

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


def mkbounds(labels,initpars=None):
    """
    Make upper and lower bounds for Payne labels.

    Parameters
    ----------
    labels : list
        List of parameter names.
    initpars : numpy array, optional
        Input list of initial label guesses.  Optional

    Returns
    -------
    bounds : tuple
       Two-element tuple of lower and upper boundaries for the input labels.

    Example
    -------
    .. code-block:: python

         bounds = mkbounds(labels,initpars)

    """

    if initpars is None:
        initpars = mkinitlabels(labels)
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
        if par.upper()=='VMICRO' or par.upper()=='VTURB':
            lbounds[i] = np.maximum(initpars[i]-2,0)
            ubounds[i] = initpars[i]+2
        if par.upper().endswith('_H'):
            lbounds[i] = np.maximum(initpars[i]-0.75,-2.5)
            ubounds[i] = np.minimum(initpars[i]+0.75,0.5)
        if par.upper()=='FE_H':
            lbounds[i] = -2.5
            ubounds[i] = 0.5
        if par.upper()=='VSINI' or par.upper()=='VROT':
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


class PayneModel(object):
    """
    A class to represent a Payne Artificial Neural Network model.

    Parameters
    ----------
    coeffs : list
        List of Payne coefficient arrays.
    wavelength : numpy array
        Array of wavelength values.
    labels : list
        List of Payne labels.
    wavevac : boolean, optional
        Whether wavelengths are in vacuum units.  Default is False.

    """
    
    def __init__(self,coeffs,wavelength,labels,wavevac=False):
        """ Initialize PayneModel object. """
        self._coeffs = coeffs
        self._dispersion = wavelength
        self.labels = list(labels)
        self._wavevac = wavevac
        wr = np.zeros(2,np.float64)
        wr[0] = np.min(wavelength)
        wr[1] = np.max(wavelength)
        self.wr = wr
        self.npix = len(self._dispersion)
        self._lsf = None
        
    @property
    def dispersion(self):
        """ Wavelength array."""
        return self._dispersion

    @dispersion.setter
    def dispersion(self,disp):
        if len(disp) != len(self._dispersion):
            raise ValueError('Input dispersion array not of the right length')
        self._dispersion = disp
    
    @property
    def wavevac(self):
        """ Whether wavelengths are in vacuum units."""
        return self._wavevac
    
    @wavevac.setter
    def wavevac(self,wavevac):
        """ Set wavelength wavevac value."""
    
        # Convert wavelength from air->vacuum or vice versa
        if self._wavevac != wavevac:
            # Air -> Vacuum
            if wavevac is True:
                self._dispersion = astro.airtovac(self._dispersion)
                self._wavevac = True
            # Vacuum -> Air
            else:
                self._dispersion = astro.vactoair(self._dispersion)
                self._wavevac = False
        self.wr[0] = np.min(self._dispersion)
        self.wr[1] = np.max(self._dispersion)        


    def __call__(self,labels,spec=None,wr=None,rv=None,vsini=None,vmacro=None,fluxonly=False,wave=None):
        """
        Create the Payne model spectrum given the input label values.

        Parameters
        ----------
        labels : list or array
            List or Array of input labels values to use.
        spec : Spec1D object, optional
            Observed spectrum to use for LSF convolution and wavelength array.  Default is to return
            the full model spectrum with no convolution.
        wr : list or array, optional
            Two-element list or array giving the upper and lower wavelength ranges for the output
            model spectrum.
        rv : float, optional
            Doppler shift to apply to the Payne model (in km/s).  Default is no Doppler shift.
        vsini : float, optional
            Rotational broadening to apply to the Payne model (in km/s).  Default is no rotational
            broadening.
        vmacro : float, optional
            Extra Gaussian broadening to apply to Payne model (in km/s) for macroturbulence.
            Default is no Gaussian broadening.
        fluxonly : boolean, optional
            Only return the flux array.  Default is to return a Spec1D object.
        wave : numpy array, optional
            Input wavelength array to use for the output Payne model.  Default is to use the
            observed spectrum wavelengths.

        Returns
        -------
        mspec : numpy array or Spec1D object
            The output model Payne spectrum.  If fluxonly=True then only the flux array is returned,
            otherwise a Spec1D object is returned.

        Example
        -------
        .. code-block:: python

             mspec = model(labels)

        """
        
        if len(labels) != len(self.labels):
            raise ValueError('labels must have '+str(len(self.labels))+' elements')
        
        # Prepare the spectrum
        if spec is not None:
            out = self.prepare(labels,spec=spec,rv=rv,vsini=vsini,vmacro=vmacro,wave=wave)
            if fluxonly is True:
                return out.flux
            return out

        '''
        Predict the rest-frame spectrum (normalized) of a single star.
        We input the scaled stellar labels (not in the original unit).
        Each label ranges from -0.5 to 0.5
        '''

        # Input wavelengths, create WR 
        if wave is not None:
            wr = [np.min(wave),np.max(wave)]
            if rv is not None:
                wr = [np.min([wr[0],wr[0]*(1+rv/cspeed)]), np.max([wr[1],wr[1]*(1+rv/cspeed)])]
            
        # assuming your NN has two hidden layers.
        w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max = self._coeffs
        scaled_labels = (labels-x_min)/(x_max-x_min) - 0.5
        inside = np.einsum('ij,j->i', w_array_0, scaled_labels) + b_array_0
        outside = np.einsum('ij,j->i', w_array_1, leaky_relu(inside)) + b_array_1
        spectrum = np.einsum('ij,j->i', w_array_2, leaky_relu(outside)) + b_array_2
        wavelength = self._dispersion.copy()
        
        # Trim
        if wr is not None:
            gd, = np.where( (self.dispersion >= wr[0]) & (self.dispersion <= wr[1]) )
            if len(gd)==0:
                raise Exception('No pixels between '+str(wr[0])+' and '+str(wr[1]))
            spectrum = spectrum[gd]
            wavelength = wavelength[gd]

        # Apply Vmacro broadening and Vsini broadening (km/s)
        if (vmacro is not None) | (vsini is not None):
            spectrum = utils.broaden(wavelength,spectrum,vgauss=vmacro,vsini=vsini)

        # Interpolate onto a new wavelength scale
        if (rv is not None and rv != 0.0) or wave is not None:
            inspectrum = spectrum
            inwave = wavelength
            outwave = wavelength
            
            # Apply radial velocity to input wavelength scale
            if (rv is not None and rv != 0.0):
                inwave *= (1+rv/cspeed)

            # Use WAVE for output wavelengths
            if wave is not None:
                # Currently this only handles 1D wavelength arrays                
                outwave = wave.copy()
                
            # Do the interpolation
            spectrum = interp1d(inwave,inspectrum,kind='cubic',bounds_error=False,
                                fill_value=(np.nan,np.nan),assume_sorted=True)(outwave)
            wavelength = outwave

        # Return as spectrum object with wavelengths
        if fluxonly is False:
            mspec = Spec1D(spectrum,wave=wavelength,lsfsigma=None,instrument='Model')
        else:
            mspec = spectrum
                
        return mspec

    
    def label_arrayize(self,labeldict):
        """
        Convert labels from a dictionary or numpy structured array to array.

        Parameters
        ----------
        labeldict : dictionary
            Dictionary of label values.  Values for all model labels need to be given.

        Returns
        -------
        arr : numpy array
            Array of label values.
        
        Example
        -------
        .. code-block:: python

             labelarr = model.label_arrayize(labeldict)

        """
        arr = np.zeros(len(self.labels),np.float64)
        for i in range(len(self.labels)):
            val = labeldict.get(self.labels[i])
            if val == None:
                raise ValueError(self.labels[i]+' NOT FOUND')
            arr[i] = val
        return arr
    

    def copy(self):
        """ Make a full copy of the PayneModel object. """
        new_coeffs = []
        for c in self._coeffs:
            new_coeffs.append(c.copy())
        new = PayneModel(new_coeffs,self._dispersion.copy(),self.labels.copy())
        return new

    
    @classmethod
    def read(cls,mfile):
        """ Read in a single Payne Model."""
        coeffs, wavelength, labels, wavevac = load_payne_model(mfile)
        return PayneModel(coeffs, wavelength, labels, wavevac=wavevac)


    def prepare(self,labels,spec,rv=None,vmacro=None,vsini=None,wave=None):
        """
        Prepare a Payne model spectrum using an observed spectrum.

        Parameters
        ----------
        labels : list or array
            List or Array of input labels values to use.
        spec : Spec1D object, optional
            Observed spectrum to use for LSF convolution and wavelength array.  Default is to return
            the full model spectrum with no convolution.
        rv : float, optional
            Doppler shift to apply to the Payne model (in km/s).  Default is no Doppler shift.
        vmacro : float, optional
            Extra Gaussian broadening to apply to Payne model (in km/s) for macroturbulence.
            Default is no Gaussian broadening.
        vsini : float, optional
            Rotational broadening to apply to the Payne model (in km/s).  Default is no rotational
            broadening.
        wave : numpy array, optional
            Input wavelength array to use for the output Payne model.  Default is to use the
            observed spectrum wavelengths.

        Returns
        -------
        mspec : Spec1D object
            The output model Payne Spec1D spectrum.

        Example
        -------
        .. code-block:: python

             mspec = model.prepare(labels,spec)

        """

        return prepare_payne_model(self,labels,spec,rv=rv,vmacro=vmacro,vsini=vsini,wave=wave)


        
class PayneModelSet(object):
    """
    A class to represent a set of Payne Artificial Neural Network models.  This is used
    when separate Payne models are used to cover a different "chunk" of wavelength.

    Parameters
    ----------
    models : list of PayneModel objects
        List of PayneModel objects.

    """
    
    def __init__(self,models):
        """ Initialize PayneModel object. """
        # Make sure it's a list
        if type(models) is not list:
            models = [models]
        # Check that the input is Payne models
        if not isinstance(models[0],PayneModel):
            raise ValueError('Input must be list of Payne models')
            
        self.nmodel = len(models)
        self._data = models
        wrarray = np.zeros((2,len(models)),np.float64)
        disp = []
        for i in range(len(models)):
            wrarray[0,i] = np.min(models[i].dispersion)
            wrarray[1,i] = np.max(models[i].dispersion)
            disp += list(models[i].dispersion)
        self._wrarray = wrarray
        self._dispersion = np.array(disp)

        self._wavevac = self._data[0]._wavevac
        self.npix = len(self._dispersion)
        wr = np.zeros(2,np.float64)
        wr[0] = np.min(self._dispersion)
        wr[1] = np.max(self._dispersion)
        self.wr = wr   # global wavelength range
        self.labels = self._data[0].labels
        self._lsf = None
        
    @property
    def dispersion(self):
        """ Wavelength array."""
        return self._dispersion
    
    @property
    def wavevac(self):
        """ Whether wavelengths are in vacuum units."""
        return self._wavevac
    
    @wavevac.setter
    def wavevac(self,wavevac):
        """ Set wavelength wavevac value."""
    
        # Convert wavelength from air->vacuum or vice versa
        if self._wavevac != wavevac:
            wrarray = np.zeros((2,self.nmodel),np.float64)
            disp = np.zeros(self.npix,np.float64)
            count = 0
            for i in range(self.nmodel):
                self._data[i].wavevac = wavevac    # convert the chunk model
                wrarray[:,i] = self._data[i].wr
                disp[count:count+self._data[i].npix] = self._data[i]._dispersion
                count += self._data[i].npix
            self._wrarray = wrarray
            self._dispersion = disp
            self._wavevac = wavevac
            # Recalculate global wavelength range
            wr = np.zeros(2,np.float64)
            wr[0] = np.min(wrarray)
            wr[1] = np.max(wrarray)
            self.wr = wr
    
    def __call__(self,labels,spec=None,wr=None,rv=None,vsini=None,vmacro=None,fluxonly=False,wave=None):
        """
        Create the Payne model spectrum given the input label values.

        Parameters
        ----------
        labels : list or array
            List or Array of input labels values to use.
        spec : Spec1D object, optional
            Observed spectrum to use for LSF convolution and wavelength array.  Default is to return
            the full model spectrum with no convolution.
        wr : list or array, optional
            Two-element list or array giving the upper and lower wavelength ranges for the output
            model spectrum.
        rv : float, optional
            Doppler shift to apply to the Payne model (in km/s).  Default is no Doppler shift.
        vsini : float, optional
            Rotational broadening to apply to the Payne model (in km/s).  Default is no rotational
            broadening.
        vmacro : float, optional
            Extra Gaussian broadening to apply to Payne model (in km/s) for macroturbulence.
            Default is no Gaussian broadening.
        fluxonly : boolean, optional
            Only return the flux array.  Default is to return a Spec1D object.
        wave : numpy array, optional
            Input wavelength array to use for the output Payne model.  Default is to use the
            observed spectrum wavelengths.

        Returns
        -------
        mspec : numpy array or Spec1D object
            The output model Payne spectrum.  If fluxonly=True then only the flux array is returned,
            otherwise a Spec1D object is returned.

        Example
        -------
        .. code-block:: python

             mspec = model(labels)

        """

        
        '''
        Predict the rest-frame spectrum (normalized) of a single star.
        We input the scaled stellar labels (not in the original unit).
        Each label ranges from -0.5 to 0.5
        '''

        if len(labels) != len(self.labels):
            raise ValueError('labels must have '+str(len(self.labels))+' elements')

        # Prepare the spectrum
        if spec is not None:
            out = self.prepare(labels,spec=spec,rv=rv,vsini=vsini,vmacro=vmacro,wave=wave)
            if fluxonly is True:
                return out.flux
            return out

        # Input wavelengths, create WR 
        if wave is not None:
            wr = [np.min(wave),np.max(wave)]
            if rv is not None:
                wr = [np.min([wr[0],wr[0]*(1+rv/cspeed)]), np.max([wr[1],wr[1]*(1+rv/cspeed)])]
                
        # Only a subset of wavelenths requested
        if wr is not None:
            # Check that we have pixels in this range
            lo, = np.where(self._dispersion >= wr[0])
            hi, = np.where(self._dispersion <= wr[1])          
            if (len(lo)==0) | (len(hi)==0):
                raise Exception('No pixels between '+str(wr[0])+' and '+str(wr[1]))
            # Get the chunks that we need
            #gg, = np.where( (self._wrarray[0,:] >= wr[0]) & (self._wrarray[1,:] <= wr[1]) )
            gg, = np.where( ((self._wrarray[1,:] >= wr[0]) & (self._wrarray[1,:] <= wr[1])) |
                            ((self._wrarray[0,:] <= wr[1]) & (self._wrarray[0,:] >= wr[0])) )
            ngg = len(gg)
            npix = 0
            for i in range(ngg):
                npix += self._data[gg[i]].npix
            spectrum = np.zeros(npix,np.float64)
            wavelength = np.zeros(npix,np.float64)
            cnt = 0
            for i in range(ngg):
                spec1 = self._data[gg[i]](labels,fluxonly=True)
                nspec1 = len(spec1)
                spectrum[cnt:cnt+nspec1] = spec1
                wavelength[cnt:cnt+nspec1] = self._data[gg[i]].dispersion               
                cnt += nspec1
            # Now trim a final time
            ggpix, = np.where( (wavelength >= wr[0]) & (wavelength <= wr[1]) )
            wavelength = wavelength[ggpix]
            spectrum = spectrum[ggpix]

        # all pixels
        else:
            spectrum = np.zeros(self.npix,np.float64)
            cnt = 0
            for i in range(self.nmodel):
                spec1 = self._data[i](labels,fluxonly=True)
                nspec1 = len(spec1)
                spectrum[cnt:cnt+nspec1] = spec1
                cnt += nspec1
            wavelength = self._dispersion.copy()

        # Apply Vmacro broadening and Vsini broadening (km/s)
        if (vmacro is not None) | (vsini is not None):
            spectrum = utils.broaden(wavelength,spectrum,vgauss=vmacro,vsini=vsini)

        # Interpolate onto a new wavelength scale
        if (rv is not None and rv != 0.0) or wave is not None:
            inspectrum = spectrum
            inwave = wavelength
            outwave = wavelength
            
            # Apply radial velocity to input wavelength scale
            if (rv is not None and rv != 0.0):
                inwave *= (1+rv/cspeed)

            # Use WAVE for output wavelengths
            if wave is not None:
                # Currently this only handles 1D wavelength arrays                
                outwave = wave.copy()
                
            # Do the interpolation
            spectrum = interp1d(inwave,inspectrum,kind='cubic',bounds_error=False,
                                fill_value=(np.nan,np.nan),assume_sorted=True)(outwave)
            wavelength = outwave

        # Change single order 2D arrays to 1D
        if spectrum.ndim==1:
            spectrum = spectrum.flatten()
            wavelenth = wavelength.flatten()
            
        # Return as spectrum object with wavelengths
        if fluxonly is False:
            mspec = Spec1D(spectrum,wave=wavelength,lsfsigma=None,instrument='Model')
        else:
            mspec = spectrum
                
        return mspec   

    def __setitem__(self,index,data):
        self._data[index] = data
    
    def __getitem__(self,index):
        # Return one of the Payne models in the set
        return self._data[index]

    def __len__(self):
        return self.nmodel
    
    def __iter__(self):
        self._count = 0
        return self
        
    def __next__(self):
        if self._count < self.nmodel:
            self._count += 1            
            return self._data[self._count-1]
        else:
            raise StopIteration

    def copy(self):
        """ Make a copy of the PayneModelSet."""
        new_models = []
        for d in self._data:
            new_models.append(d.copy())
        new = PayneModelSet(new_models)
        return new

    @classmethod
    def read(cls,mfiles):
        """ Read a set of Payne model files."""
        n = len(mfiles)
        models = []
        for i in range(n):
            models.append(PayneModel.read(mfiles[i]))
        # Sort by wavelength
        def minwave(m):
            return m.dispersion[0]
        models.sort(key=minwave)
        return PayneModelSet(models)

    def prepare(self,labels,spec,rv=None,vmacro=None,vsini=None,wave=None):
        """
        Prepare a Payne model spectrum using an observed spectrum.

        Parameters
        ----------
        labels : list or array
            List or Array of input labels values to use.
        spec : Spec1D object, optional
            Observed spectrum to use for LSF convolution and wavelength array.  Default is to return
            the full model spectrum with no convolution.
        rv : float, optional
            Doppler shift to apply to the Payne model (in km/s).  Default is no Doppler shift.
        vmacro : float, optional
            Extra Gaussian broadening to apply to Payne model (in km/s) for macroturbulence.
            Default is no Gaussian broadening.
        vsini : float, optional
            Rotational broadening to apply to the Payne model (in km/s).  Default is no rotational
            broadening.
        wave : numpy array, optional
            Input wavelength array to use for the output Payne model.  Default is to use the
            observed spectrum wavelengths.

        Returns
        -------
        mspec : Spec1D object
            The output model Payne Spec1D spectrum.

        Example
        -------
        .. code-block:: python

             mspec = model.prepare(labels,spec)

        """
        
        return prepare_payne_model(self,labels,spec,rv=rv,vmacro=vmacro,vsini=vsini,wave=wave)


    
class DopplerPayneModel(object):
    """
    A class to thinly wraps the PayneModel or PayneModelSet so that VSINI, VMACRO, and RV
    can be used as labels as well.  This is the primary Payne class to use with Doppler.

    Parameters
    ----------
    models : PayneModel or PayneModelSet object
        PayneModel or PayneModelSet object
    """

    def __init__(self,model):
        """ Initialize PayneModel object. """
        # Make sure it's a PayneModel or PayneModelSet
        if (not isinstance(model,PayneModel)) & (not isinstance(model,PayneModelSet)):
            raise ValueError("Model must be either PayneModel or PayneModelSet")
        self._data = model
        labels = list(model.labels.copy())
        # Add vsini, vmacro, rv
        labels += ['VSINI','VMACRO','RV']
        self.labels = labels
        self._spec = None
        self._prepared = False
        self._dispersion = self._data.dispersion
        self._wavevac = self._data._wavevac
        self._original_wavevac = self._data._wavevac
        self.wr = self._data.wr
        self._data._lsf = None

        
    def __call__(self,labels,wr=None,wave=None):
        """
        Create the Payne model spectrum given the input label values.

        Parameters
        ----------
        labels : list or array
            List/array or dictionary of input labels values to use.
        wr : list or array, optional
            Two-element list or array giving the upper and lower wavelength ranges for the output
            model spectrum.
        wave : numpy array, optional
            Input wavelength array to use for the output Payne model.  Default is to use the full
            wavelength range or the observed spectrum wavelengths if the model is "prepared".

        Returns
        -------
        mspec : numpy array or Spec1D object
            The output model Payne spectrum.  If fluxonly=True then only the flux array is returned,
            otherwise a Spec1D object is returned.

        Example
        -------
        .. code-block:: python

             mspec = model(labels)

        """
        # Dictionary input
        if isinstance(labels,dict):
            labels = self.mklabels(labels)  # convert dictionary to array of labels
            
        if len(labels) != len(self.labels):
            raise ValueError('labels must have '+str(len(self.labels))+' elements')
        vsini,vmacro,rv = labels[-3:]
        plabels = labels[0:-3]  # just the payne labels
        if self.prepared==True:
            return self._data(plabels,spec=self._spec,vsini=vsini,vmacro=vmacro,rv=rv,wave=wave,wr=wr)
        else:
            return self._data(plabels,vsini=vsini,vmacro=vmacro,rv=rv,wave=wave,wr=wr)

        
    def mklabels(self,inputs):
        """
        Convert input dictionary to labels.  Not all labels need to be specified.

        Parameters
        ----------
        inputs : dict
            Dictionary of label values. Not all labels need to be specified.
            Unspecified labels will be determined from the inputs (e.g. FE_H and
            ALPHA_H set elements) or default values.

        Returns
        -------
        labels : numpy array
            Array of label values.

        Example
        -------
        .. code-block:: python

             labels = model.mklabels(labeldict)

        """

        # This assumes ALL abundances are relative to H *not* FE!!!
        
        params = dict((key.upper(), value) for (key, value) in inputs.items()) # all CAPS
        nparams = len(params)

        labelnames = np.char.array(self.labels)
        
        # Minimum required inputs, TEFF, LOGG, FE_H
        minlabels = ['TEFF','LOGG','FE_H']
        for f in minlabels:
            if f not in params.keys():
                raise ValueError(f+' is a required input parameter')

        # Initializing the labels array
        nlabels = len(self.labels)
        #labels = self.initpars()
        labels = np.zeros(nlabels,float)
        # Set X_H = FE_H
        labels[labelnames.endswith('_H')] = params['FE_H']
        # Vmicro/Vturb=2.0 km/s by default
        labels[(labelnames=='VTURB') | (labelnames=='VMICRO')] = 2.0
        
        # Deal with alpha abundances
        # Individual alpha elements will overwrite the mean alpha below     
        # Make sure ALPHA_H is *not* one of the labels:
        if 'ALPHA_H' not in self.labels:
            if 'ALPHA_H' in params.keys():
                alpha = params['ALPHA_H']
                alphaelem = ['O','MG','SI','S','CA','TI']                
                for k in range(len(alphaelem)):
                    # Only set the value if it was found in self.labels
                    labels[labelnames==alphaelem[k]+'_H'] = alpha
                
        # Loop over input parameters
        for name in params.keys():
            # Only set the value if it was found in self.labels
            labels[labelnames==name] = params[name]
            
        return labels
                
    @property
    def dispersion(self):
        """ Wavelength array."""
        return self._dispersion
        
    @property
    def wavevac(self):
        """ Whether wavelengths are in vacuum units."""
        return self._data.wavevac
    
    @wavevac.setter
    def wavevac(self,wavevac):
        """ Set wavelength wavevac value."""
        self._data.wavevac = wavevac

    @property
    def prepared(self):
        """ Has the model been prepared with an observed spectrum."""
        return self._prepared

    @prepared.setter
    def prepared(self,value):
        """ Can be used to unprepare() the model."""
        if (value==0) | (value==False):
            self.unprepare()

    def unprepare(self):
        """ Used to unprepare the model."""
        self._spec = None
        self._prepared = False
        self._dispersion = self._data.dispersion   
        self._data.wavevac = self._original_wavevac  # reset to original
        self._wavevac = self._original_wavevac
        self.wr = self._data.wr
        self._data._lsf = None
        
    def prepare(self,spec):
        """
        Prepare the model using an observed spectrum.  This keeps a copy of the spectrum,
        makes sure the model wavevac matches the observed spectrum value, and obtains
        the 2D LSF array for the model to use later on.

        After a DopplerPayneModel has been prepared, then using __call__ will automatically
        return a Payne model spectrum that has been convolved with the observed spectrum's
        LSF and on its wavelength array.

        NOTE that this prepare() method is *very* different from the PayneModel and PayneModelSet
        prepare() method which actually returns Payne model spectrum convolved with the LSF
        of an observed spectrum.  The DopplerPayneModel prepare() is designed to behave in a
        synonymous manner to the DopplerCannonModel class.

        Parameters
        ----------
        spec : Spec1D object
            The observed spectrum to prepare with.

        Returns
        -------
        The model object is modified in place.

        Example
        -------

        .. code-block:: python
             
             model.prepare(spec)

        """
        self._spec = spec.copy()
        self._prepared = True
        self._dispersion = spec.wave.copy()
        self._data.wavevac = spec.wavevac   # switch air<->vac if necessary
        self._wavevac = spec.wavevac
        self.wr = dln.minmax(self._dispersion)
        # Get the LSF array
        tlabels = self.mklabels({'teff':5000,'logg':3.0,'fe_h':0.0})
        tlabels = tlabels[0:33]
        temp,lsf,lsfwave = prepare_payne_model(self._data,tlabels,self._spec,lsfout=True)
        self._data._lsf = lsf
        self._data._lsfwave = lsfwave
        
    def copy(self):
        """ Make a copy of the DopplerPayneModel but point to the original data."""
        new_model = self._data.copy()
        new = DopplerPayneModel(new_model)
        # points to the ORIGINAL Payne data to save space        
        if isinstance(new_model,PayneModel):
            new._data = self._data
        if isinstance(new_model,PayneModelSet):
            new._data._data = self._data._data
        if self.prepared==True:
            new.prepare(self._spec)
        return new

    def hardcopy(self):
        """ Make a complete copy of the DopplerPayneModel including the original data."""
        new_model = self._data.copy()
        new = DopplerPayneModel(new_model)
        if self.prepared==True:
            new.prepare(self._spec)
        return new    

    @classmethod
    def read(cls,mfiles):
        """ Read a set of Payne model files."""
        if dln.size(mfiles)==1:
            model = PayneModel.read(mfiles)
        else:
            model = PayneModelSet.read(mfiles)
        return DopplerPayneModel(model)
    


class PayneSpecFitter:
    """
    This is a special class that helps with least-squares fitting of a Payne model
    to an observed spectrum using functions like curve_fit().

    Parameters
    ----------
    spec : Spec1D object
       Observed spectrum to fit.
    pmodel : DopplerPayneModel object
       The DopplerPayneModel object to use for the model.
    fitparams : list, optional
       List of label names to fit.  Default is to fit all labels.
    fixparams : dictionary, optional
       Dictionary of parameter values to hold fixed.  Default is to not hold
       any values fixed.
    verbose : boolean, optional
       Verbose output to the screen.  Default is False.
    """
    
    def __init__(self,spec,pmodel,fitparams=None,fixparams={},verbose=False):
        """ Initialize PayneSpecFitter object."""
        # spec - observed spectrum object
        # pmodel - Payne model object
        # params - initial/fixed parameters dictionary
        # fitparams - parameter/label names to fit (default is all)
        # "Prepare" the Payne model with the observed spectrum
        if pmodel.prepared is False: pmodel.prepare(spec)
        self._paynemodel = pmodel
        self.labels = pmodel.labels
        labelnames = np.char.array(self._paynemodel.labels)
        nlabels = len(self._paynemodel.labels)
        self.fixparams = dict((key.upper(), value) for (key, value) in fixparams.items()) # all CAPS
        self._initlabels = self.mkinitlabels(fixparams)
        if fitparams is not None:
            self.fitparams = fitparams
        else:
            self.fitparams = paynemodel.labels # by default fit all Payne parameters
        self._nfit = len(self.fitparams)

        # Labels FIXED, ALPHAELEM, ELEM arrays
        fixed = np.ones(nlabels,bool)  # all fixed by default
        alphaelem = np.zeros(nlabels,bool)  # all False to start
        elem = np.zeros(nlabels,bool)  # all False to start        
        for k,name in enumerate(labelnames):
            # Alpha element
            if name in ['O_H','MG_H','SI_H','S_H','CA_H','TI_H']:
                alphaelem[k] = True
                elem[k] = True
                # In FITPARAMS, NOT FIXED
                if name in self.fitparams:
                    fixed[k] = False
                # Not in FITPARAMS but in FIXPARAMS, FIXED                    
                elif name in self.fixparams.keys():
                    fixed[k] = True
                # Not in FITPARAMS or FIXPARAMS, but FE_H or ALPHA_H in FITPARAMS, NOT FIXED
                elif 'FE_H' in self.fitparams or 'ALPHA_H' in self.fitparams:
                    fixed[k] = False
                # Not in FITPARAMS/PARAMS and FE_H/ALPHA_H not being fit, FIXED
                else:
                    fixed[k] = True
            # Non-alpha element
            elif name.endswith('_H'):
                elem[k] = True
                # In FITPARAMS, NOT FIXED
                if name in self.fitparams:
                    fixed[k] = False
                # Not in FITPARAMS but in FIXPARAMS, FIXED
                elif name in self.fixparams.keys():
                    fixed[k] = True
                # Not in FITPARAMS or FIXPARAMS, but FE_H in FITPARAMS, NOT FIXED
                elif 'FE_H' in self.fitparams:
                    fixed[k] = False
                # Not in FITPARAMS/FIXPARAMS and FE_H not being fit, FIXED
                else:
                    fixed[k] = True
            # Other parameters (Teff, logg, RV, Vturb, Vsini, etc.)
            else:
                # In FITPARAMS, NOT FIXED
                if name in self.fitparams:
                    fixed[k] = False
                # Not in FITPARAMS but in FIXPARAMS, FIXED
                elif name in self.fixparams.keys():
                    fixed[k] = True
                # Not in FITPARAMS/PARAMS, FIXED
                else:
                    fixed[k] = True
        self._label_fixed = fixed
        self._label_alphaelem = alphaelem
        self._label_elem = elem        

        self._spec = spec.copy()
        self._flux = spec.flux.flatten()
        self._err = spec.err.flatten()
        self._wave = spec.wave.flatten()
        self._lsf = spec.lsf.copy()
        self._lsf.wavevac = spec.wavevac
        self._wavevac = spec.wavevac
        self.verbose = verbose
        #self._norm = norm   # normalize
        self._continuum_func = spec.continuum_func
        # Figure out the wavelength parameters
        npix = spec.npix
        norder = spec.norder
        # parameters to save
        self.nfev = 0
        self.njac = 0
        self._all_pars = []
        self._all_model = []
        self._all_chisq = []
        self._jac_array = None

    @property
    def fixparams(self):
        """ Dictionary of fixed parameters."""
        return self._fixparams

    @fixparams.setter
    def fixparams(self,fixparams):
        """ Dictionary, keys must be all CAPS."""
        self._fixparams = dict((key.upper(), value) for (key, value) in fixparams.items())  # all CAPS
            
    @property
    def fitparams(self):
        """ List of labels to fit."""
        return self._fitparams

    @fitparams.setter
    def fitparams(self,fitparams):
        """ List, all CAPS."""
        self._fitparams = [f.upper() for f in fitparams]

    def mkinitlabels(self,inputs):
        """
        Make initial guesses for Payne labels.

        Parameters
        ----------
        inputs : dict
           Dictionary of parameter values to use in the array.

        Returns
        -------
        labels : numpy array
           Array of initial label values.

        Example
        -------
        .. code-block:: python

             labels = spfitter.mkinitlabels(labeldict)
        
        """
        
        # This assumes ALL abundances are relative to H *not* FE!!!
        
        params = dict((key.upper(), value) for (key, value) in inputs.items()) # all CAPS
        nparams = len(params)

        labelnames = np.char.array(self._paynemodel.labels)

        # Defaults for main parameters
        if 'TEFF' not in list(params.keys()):
            params['TEFF'] = 4000.0
        if 'LOGG' not in list(params.keys()):
            params['LOGG'] = 3.0
        if 'FE_H' not in list(params.keys()):
            params['FE_H'] = 0.0            
            
            
        # Initializing the labels array
        nlabels = len(self._paynemodel.labels)
        labels = np.zeros(nlabels,float)
        # Set X_H = FE_H
        labels[labelnames.endswith('_H')] = params['FE_H']
        # Vmicro/Vturb=2.0 km/s by default
        labels[(labelnames=='VTURB') | (labelnames=='VMICRO')] = 2.0
        
        # Deal with alpha abundances
        # Individual alpha elements will overwrite the mean alpha below     
        # Make sure ALPHA_H is *not* one of the labels:
        if 'ALPHA_H' not in self._paynemodel.labels:
            if 'ALPHA_H' in params.keys():
                alpha = params['ALPHA_H']
                alphaelem = ['O','MG','SI','S','CA','TI']                
                for k in range(len(alphaelem)):
                    # Only set the value if it was found in self.labels
                    labels[labelnames==alphaelem[k]+'_H'] = alpha
                
        # Loop over input parameters
        for name in params.keys():
            # Only set the value if it was found in labelnames
            labels[labelnames==name] = params[name]
            
        return labels

        
    def mklabels(self,args):
        """
        Make labels for Payne model using values for only the fitted values.

        Parameters
        ----------
        args : list or tuple
           List or tuple of values for the fitted parameters (fitparams).

        Returns
        -------
        labels : numpy array
           Array of values for all the Payne labels.

        Example
        -------
        .. code-block:: python

             labels = spfitter.mklabels(args)
              
        """
        # Start with initial labels and only modify the fitparams."""

        # Initialize with init values
        labels = self._initlabels.copy()
        
        labelnames = np.char.array(self._paynemodel.labels)
        fitnames = np.char.array(self.fitparams)
        if 'FE_H' in self.fitparams:
            fitfeh = True
            fehind, = np.where(fitnames=='FE_H')
        else:
            fitfeh = False
        if 'ALPHA_H' in self.fitparams:
            fitalpha = True
            alphaind, = np.where(fitnames=='ALPHA_H')
        else:
            fitalpha = False

        # Loop over labels        
        for k,name in enumerate(labelnames):
            # Label is NOT fixed, change it
            if self._label_fixed[k] == False:
                # Alpha element
                if self._label_alphaelem[k] == True:
                    # ALPHA_H in FITPARAMS
                    if fitalpha is True:
                        labels[k] = args[alphaind[0]]
                    elif fitfeh is True:
                        labels[k] = args[fehind[0]]
                    else:
                        print('THIS SHOULD NOT HAPPEN!')
                        import pdb; pdb.set_trace()
                # Non-alpha element
                elif self._label_elem[k] == True:
                    if fitfeh is True:
                        labels[k] = args[fehind[0]]
                    else:
                        print('THIS SHOULD NOT HAPPEN!')
                        import pdb; pdb.set_trace()
                # Other parameters
                else:
                    ind, = np.where(fitnames==name)
                    labels[k] = args[ind[0]]

        # Add values for individual elements we are fitting (not FE_H or ALPHA_H)
        #   The code above misses individual elements that are being fit
        gdelem, = np.where(fitnames.endswith('_H') & (fitnames.find('FE_H')==-1) &
                           (fitnames.find('ALPHA_H')==-1))
        ngdelem = len(gdelem)
        for k in range(ngdelem):
            name = fitnames[gdelem[k]]
            ind, = np.where(labelnames==name)
            labels[ind[0]] = args[gdelem[k]]
        
        return labels
    
    def chisq(self,model):
        """
        Calculate the chi-squared between the Payne model spectrum and observed spectrum.
       
        Parameters
        ----------
        model : numpy array
           Array of Payne model spectrum flux values.

        Returns
        -------
        chisq : float
           Chi-squared value of the input Payne model spectrum and the observed spectrum.

        Example
        -------
        .. code-block:: python

             chisq = spfitter.chisq(model)

        """
        return np.sqrt( np.sum( (self._flux-model)**2/self._err**2 )/len(self._flux) )
            
    def model(self,xx,*args):
        """
        Return model Payne spectrum given the input arguments.  To be used with
        curve_fit().

        Parameters
        ----------
        xx : numpy array
            Input indepedent wavelength values.  Not used, but needed for curve_fit().
        args : tuple
            Tuple of input positional arguments of fitted model labels.

        Returns
        -------
        mflux : numpy array
            The output model Payne spectrum flux array, flattened.

        Example
        -------
        .. code-block:: python

             mflux = spfitter.model(wave,*labels)

        """        
        # Convert arguments to Payne model inputs
        labels = self.mklabels(args)
        if self.verbose: print(args)
        self.nfev += 1
        return self._paynemodel(labels).flux.flatten()  # only return the flattened flux

    def mkdxlim(self,fitparams):
        """
        Make array of parameter changes at which curve_fit should finish.

        Parameters
        ----------
        fitparams : list
            List of parameter names.

        Returns
        -------
        dx_lim : numpy array
            Array of parameter changes at which curve_fit should finish.

        Example
        -------
        .. code-block:: python

             dx_lim = spfitter.mkdxlim(fitparams)

        """
        return mkdxlim(fitparams)

    def mkbounds(self,labels,initpars=None):
        """
        Make upper and lower bounds for Payne labels.
        
        Parameters
        ----------
        labels : list
            List of parameter names.
        initpars : numpy array, optional
            Input list of initial label guesses.  Optional

        Returns
        -------
        bounds : tuple
            Two-element tuple of lower and upper boundaries for the input labels.

        Example
        -------
        .. code-block:: python
        
             bounds = spfitter.mkbounds(labels,initpars)

        """
        return mkbounds(labels,initpars=initpars)

    def getstep(self,name,val=None,relstep=0.02):
        """
        Calculate step for a single parameter to be used to generate the Jacobian.
        """
        if name=='TEFF':
            step = 5.0
        elif name=='RV':
            step = 0.1
        elif (name=='VROT' or name=='VSINI'):
            step = 1.0
        elif (name=='VMICRO' or name=='VTURB'):
            step = 0.25
        elif name=='VMACRO':
            step = 0.25
        elif name.endswith('_H'):
            step = 0.01
        else:
            step = 0.01
        return step
                
    def jac(self,x,*args):
        """
        Compute the Jacobian matrix (an Npix-by-Npar matrix, where element (i, j)
        is the partial derivative of f[i] with respect to x[j]).  This is to be
        used with curve_fit().

        Parameters
        ----------
        args : tuple
            Tuple of input positional arguments of fitted model labels at which
            to calculate the Jacobian.

        Returns
        -------
        jac : numpy array
           Jacobian matrix (an Npix-by-Npar matrix) of how the model changes
           (at each pixel) with respect to each parameter.

        Example
        -------
         .. code-block:: python
              
              jac = spfitter.jac(wave,*args)

        """
        
        # Boundaries
        lbounds,ubounds = self.mkbounds(self.fitparams)
        
        relstep = 0.02
        npix = len(x)
        npar = len(args)
        
        # Create synthetic spectrum at current values
        f0 = self.model(self._wave,*args)
        #self.nfev += 1
        
        # Save models/pars/chisq
        self._all_pars.append(list(args).copy())
        self._all_model.append(f0.copy())
        self._all_chisq.append(self.chisq(f0))
        chisq = np.sqrt( np.sum( (self._flux-f0)**2/self._err**2 )/len(self._flux) )
        #if self.verbose:
        #    print('chisq = '+str(chisq))
        
        # Initialize jacobian matrix
        jac = np.zeros((npix,npar),np.float64)
        
        # Loop over parameters
        for i in range(npar):
            pars = np.array(copy.deepcopy(args))
            step = self.getstep(self.fitparams[i],pars[i],relstep)
            # Check boundaries, if above upper boundary
            #   go the opposite way
            if pars[i]>ubounds[i]:
                step *= -1
            pars[i] += step
            
            #if self.verbose:
            #    print('--- '+str(i+1)+' '+self.fitparams[i]+' '+str(pars[i])+' ---')

            f1 = self.model(self._wave,*pars)
            #self.nfev += 1
            
            # Save models/pars/chisq
            self._all_pars.append(list(pars).copy())
            self._all_model.append(f1.copy())
            self._all_chisq.append(self.chisq(f1))
            
            if np.sum(~np.isfinite(f1))>0:
                print('some nans/infs')
                import pdb; pdb.set_trace()

            jac[:,i] = (f1-f0)/step
            
        if np.sum(~np.isfinite(jac))>0:
            print('some nans/infs')
            import pdb; pdb.set_trace()

        self._jac_array = jac.copy()   # keep a copy
            
        self.njac += 1
            
        return jac

    
class PayneMultiSpecFitter:
    """
    This is a special class that helps with least-squares fitting of a Payne model
    to multiple observed spectra using functions like curve_fit().

    Parameters
    ----------
    speclist : list of Spec1D objects
       List of bbserved spectra to fit.
    modlist : list of DopplerPayneModel object
       List of DopplerPayneModel objects to use for the model for each observed spectrum.
       Each model should be prepared with their respective observed spectrum and hold
       the 2D LSF array.
    fitparams : list, optional
       List of label names to fit.  Default is to fit all labels.
    fixparams : dictionary, optional
       Dictionary of parameter values to hold fixed.  Default is to not hold
       any values fixed.
    verbose : boolean, optional
       Verbose output to the screen.  Default is False.
    """
    
    def __init__(self,speclist,modlist,fitparams,fixparams={},verbose=False):
        """ Initialize PayneSpecFitter object."""        
        # speclist - observed spectrum list
        # modlist - Payne model list
        # fixparams - parameter/label names to fit
        # fitparams - fixed parameters dictionary
        self.nspec = len(speclist)
        self._speclist = speclist
        self._modlist = modlist
        self._labels = modlist[0].labels  # all labels
        self.fitparams = fitparams
        self.fixparams = dict((key.upper(), value) for (key, value) in fixparams.items()) # all CAPS
        self._initlabels = self.mkinitlabels(fixparams)
        
        # Put all of the spectra into a large 1D array
        ntotpix = 0
        for s in speclist:
            ntotpix += s.npix*s.norder
        wave = np.zeros(ntotpix)
        flux = np.zeros(ntotpix)
        err = np.zeros(ntotpix)
        cnt = 0
        for i in range(self.nspec):
            spec = speclist[i]
            npx = spec.npix*spec.norder
            wave[cnt:cnt+npx] = spec.wave.T.flatten()
            flux[cnt:cnt+npx] = spec.flux.T.flatten()
            err[cnt:cnt+npx] = spec.err.T.flatten()
            cnt += npx
        self._flux = flux
        self._err = err
        self._wave = wave        

        fixed,alphaelem,elem = self.fixed_labels(list(fitparams)+['RV'])
        self._label_fixed = fixed
        self._label_alphaelem = alphaelem
        self._label_elem = elem     
        
        self.verbose = verbose
        # parameters to save
        self.nfev = 0
        self.njac = 0
        self._all_pars = []
        self._all_model = []
        self._all_chisq = []
        self._jac_array = None

    def fixed_labels(self,fitparams=None):
        """ Helper function to determine which labels are fixed."""
        nlabels = len(self._labels)
        labelnames = self._labels

        if fitparams is None:
            fitparams = self.fitparams
        
        # Labels FIXED, ALPHAELEM, ELEM arrays
        fixed = np.ones(nlabels,bool)  # all fixed by default
        alphaelem = np.zeros(nlabels,bool)  # all False to start
        elem = np.zeros(nlabels,bool)  # all False to start        
        for k,name in enumerate(labelnames):
            # Alpha element
            if name in ['O_H','MG_H','SI_H','S_H','CA_H','TI_H']:
                alphaelem[k] = True
                elem[k] = True
                # In FITPARAMS, NOT FIXED
                if name in fitparams:
                    fixed[k] = False
                # Not in FITPARAMS but in FIXPARAMS, FIXED                    
                elif name in self.fixparams.keys():
                    fixed[k] = True
                # Not in FITPARAMS or FIXPARAMS, but FE_H or ALPHA_H in FITPARAMS, NOT FIXED
                elif 'FE_H' in fitparams or 'ALPHA_H' in fitparams:
                    fixed[k] = False
                # Not in FITPARAMS/PARAMS and FE_H/ALPHA_H not being fit, FIXED
                else:
                    fixed[k] = True
            # Non-alpha element
            elif name.endswith('_H'):
                elem[k] = True
                # In FITPARAMS, NOT FIXED
                if name in fitparams:
                    fixed[k] = False
                # Not in FITPARAMS but in FIXPARAMS, FIXED
                elif name in self.fixparams.keys():
                    fixed[k] = True
                # Not in FITPARAMS or FIXPARAMS, but FE_H in FITPARAMS, NOT FIXED
                elif 'FE_H' in fitparams:
                    fixed[k] = False
                # Not in FITPARAMS/FIXPARAMS and FE_H not being fit, FIXED
                else:
                    fixed[k] = True
            # Other parameters (Teff, logg, RV, Vturb, Vsini, etc.)
            else:
                # In FITPARAMS, NOT FIXED
                if name in fitparams:
                    fixed[k] = False
                # Not in FITPARAMS but in FIXPARAMS, FIXED
                elif name in self.fixparams.keys():
                    fixed[k] = True
                # Not in FITPARAMS/PARAMS, FIXED
                else:
                    fixed[k] = True
        return fixed,alphaelem,elem
        
    @property
    def fitparams(self):
        """ List of labels to fit."""
        return self._fitparams

    @fitparams.setter
    def fitparams(self,fitparams):
        """ List, all CAPS."""
        self._fitparams = [f.upper() for f in fitparams]
        
    @property
    def fixparams(self):
        """ Dictionary of fixed parameters."""
        return self._fixparams

    @fixparams.setter
    def fixparams(self,fixparams):
        """ Dictionary, keys must be all CAPS."""
        self._fixparams = dict((key.upper(), value) for (key, value) in fixparams.items())  # all CAPS

    def mkinitlabels(self,inputs):
        """
        Make initial guesses for Payne labels.

        Parameters
        ----------
        inputs : dict
           Dictionary of parameter values to use in the array.

        Returns
        -------
        labels : numpy array
           Array of initial label values.

        Example
        -------
        .. code-block:: python

             labels = spfitter.mkinitlabels(labeldict)
        
        """

        # This assumes ALL abundances are relative to H *not* FE!!!
        
        params = dict((key.upper(), value) for (key, value) in inputs.items()) # all CAPS
        nparams = len(params)

        labelnames = np.char.array(self._labels)

        # Defaults for main parameters
        if 'TEFF' not in list(params.keys()):
            params['TEFF'] = 4000.0
        if 'LOGG' not in list(params.keys()):
            params['LOGG'] = 3.0
        if 'FE_H' not in list(params.keys()):
            params['FE_H'] = 0.0            
            
        # Initializing the labels array
        nlabels = len(self._labels)
        labels = np.zeros(nlabels,float)
        # Set X_H = FE_H
        labels[labelnames.endswith('_H')] = params['FE_H']
        # Vmicro/Vturb=2.0 km/s by default
        labels[(labelnames=='VTURB') | (labelnames=='VMICRO')] = 2.0
        
        # Deal with alpha abundances
        # Individual alpha elements will overwrite the mean alpha below     
        # Make sure ALPHA_H is *not* one of the labels:
        if 'ALPHA_H' not in self._labels:
            if 'ALPHA_H' in params.keys():
                alpha = params['ALPHA_H']
                alphaelem = ['O','MG','SI','S','CA','TI']                
                for k in range(len(alphaelem)):
                    # Only set the value if it was found in self.labels
                    labels[labelnames==alphaelem[k]+'_H'] = alpha
                
        # Loop over input parameters
        for name in params.keys():
            # Only set the value if it was found in labelsnames
            labels[labelnames==name] = params[name]
            
        return labels

    def mklabels(self,args,fitparams=None):
        """
        Make labels for Payne model using values for only the fitted values.

        Parameters
        ----------
        args : list or tuple
           List or tuple of values for the fitted parameters (fitparams).

        Returns
        -------
        labels : numpy array
           Array of values for all the Payne labels.

        Example
        -------
        .. code-block:: python

             labels = spfitter.mklabels(args)
              
        """

        # Start with initial labels and only modify the fitparams."""

        # Initialize with init values
        labels = self._initlabels.copy()
        
        labelnames = np.char.array(self._labels)
        if fitparams is None:
            fitparams = self.fitparams.copy()
        fitnames = np.char.array(fitparams)
        if 'FE_H' in fitparams:
            fitfeh = True
            fehind, = np.where(fitnames=='FE_H')
        else:
            fitfeh = False
        if 'ALPHA_H' in fitparams:
            fitalpha = True
            alphaind, = np.where(fitnames=='ALPHA_H')
        else:
            fitalpha = False

        # Loop over labels        
        for k,name in enumerate(labelnames):
            # Label is NOT fixed, change it
            if self._label_fixed[k] == False:
                # Alpha element
                if self._label_alphaelem[k] == True:
                    # ALPHA_H in FITPARAMS
                    if fitalpha is True:
                        labels[k] = args[alphaind[0]]
                    elif fitfeh is True:
                        labels[k] = args[fehind[0]]
                    else:
                        print('THIS SHOULD NOT HAPPEN!')
                        import pdb; pdb.set_trace()
                # Non-alpha element
                elif self._label_elem[k] == True:
                    if fitfeh is True:
                        labels[k] = args[fehind[0]]
                    else:
                        print('THIS SHOULD NOT HAPPEN!')
                        import pdb; pdb.set_trace()
                # Other parameters
                else:
                    ind, = np.where(fitnames==name)
                    labels[k] = args[ind[0]]

        # Add values for individual elements we are fitting (not FE_H or ALPHA_H)
        #   The code above misses individual elements that are being fit
        gdelem, = np.where(fitnames.endswith('_H') & (fitnames.find('FE_H')==-1) &
                           (fitnames.find('ALPHA_H')==-1))
        ngdelem = len(gdelem)
        for k in range(ngdelem):
            name = fitnames[gdelem[k]]
            ind, = np.where(labelnames==name)
            labels[ind[0]] = args[gdelem[k]]
                    
        return labels
    
    def chisq(self,modelflux):
        """
        Calculate the chi-squared between the Payne model spectrum and observed spectrum.
       
        Parameters
        ----------
        modelflux : numpy array
           Array of Payne model spectrum flux values.

        Returns
        -------
        chisq : float
           Chi-squared value of the input Payne model spectrum and the observed spectrum.

        Example
        -------
        .. code-block:: python

             chisq = spfitter.chisq(model)

        """
        return np.sqrt( np.sum( (self._flux-modelflux)**2/self._err**2 )/len(self._flux) )
            
    def model(self,xx,*args):
        """
        Return model Payne spectrum given the input arguments.  To be used with
        curve_fit().

        Parameters
        ----------
        xx : numpy array
            Input indepedent wavelength values.  Not used, but needed for curve_fit().
        args : tuple
            Tuple of input positional arguments of fitted model labels.

        Returns
        -------
        mflux : numpy array
            The output model Payne spectrum flux array, flattened.

        Example
        -------
        .. code-block:: python

             mflux = spfitter.model(wave,*labels)

        """     
        
        # Return model Payne spectrum given the input arguments."""
        # Convert arguments to Payne model inputs
        #print(args)
        nargs = len(args)
        nfitparams = len(self.fitparams)
        params = args[0:nfitparams]
        vrel = args[nfitparams:]
        npix = len(xx)
        flux = np.zeros(npix,float)
        cnt = 0
        for i in range(self.nspec):
            npx = self._speclist[i].npix*self._speclist[i].norder
            # Create parameter list that includes RV at the end
            params1 = list(params)+[vrel[i]]
            paramnames1 = self.fitparams+['RV']
            labels = self.mklabels(params1,fitparams=paramnames1)
            m = self._modlist[i](labels)
            if m is not None:
                flux[cnt:cnt+npx] = m.flux.T.flatten()
            else:
                flux[cnt:cnt+npx] = 1e30
            cnt += npx
        self.nfev = 0
        return flux
    
    def mkdxlim(self,fitparams):
        """
        Make array of parameter changes at which curve_fit should finish.

        Parameters
        ----------
        fitparams : list
            List of parameter names.

        Returns
        -------
        dx_lim : numpy array
            Array of parameter changes at which curve_fit should finish.

        Example
        -------
        .. code-block:: python

             dx_lim = spfitter.mkdxlim(fitparams)

        """
        return mkdxlim(fitparams)

    def mkbounds(self,labels,initpars=None):
        """
        Make upper and lower bounds for Payne labels.
        
        Parameters
        ----------
        labels : list
            List of parameter names.
        initpars : numpy array, optional
            Input list of initial label guesses.  Optional

        Returns
        -------
        bounds : tuple
            Two-element tuple of lower and upper boundaries for the input labels.

        Example
        -------
        .. code-block:: python
        
             bounds = spfitter.mkbounds(labels,initpars)

        """        
        return mkbounds(labels,initpars=initpars)

    def getstep(self,name,val=None,relstep=0.02):
        """
        Calculate step for a single parameter to be used to generate the Jacobian.
        """        
        if name=='TEFF':
            step = 5.0
        elif name=='RV':
            step = 0.1
        elif name=='VROT':
            step = 0.5
        elif name=='VMICRO':
            step = 0.5
        elif name.endswith('_H'):
            step = 0.01
        else:
            step = 0.01
        return step
                
    def jac(self,x,*args):
        """
        Compute the Jacobian matrix (an Npix-by-Npar matrix, where element (i, j)
        is the partial derivative of f[i] with respect to x[j]).  This is to be
        used with curve_fit().

        Parameters
        ----------
        args : tuple
            Tuple of input positional arguments of fitted model labels at which
            to calculate the Jacobian.

        Returns
        -------
        jac : numpy array
           Jacobian matrix (an Npix-by-Npar matrix) of how the model changes
           (at each pixel) with respect to each parameter.

        Example
        -------
         .. code-block:: python
              
              jac = spfitter.jac(wave,*args)

        """

        npix = len(x)
        npar = len(args)
        nspec = len(self._speclist)
        nfitparams = npar-nspec
        
        # Boundaries
        lbounds = np.zeros(npar,float)+1e5
        ubounds = np.zeros(npar,float)-1e5
        labelbounds = self.mkbounds(self.fitparams)
        lbounds[0:nfitparams] = labelbounds[0]
        ubounds[0:nfitparams] = labelbounds[1]    
        lbounds[nfitparams:] = -1000
        ubounds[nfitparams:] = 1000    

        params = args[0:nfitparams]
        vrel = args[nfitparams:]
        # Initialize jacobian matrix
        jac = np.zeros((npix,npar),float)
        
        # Model at current values
        f0 = self.model(x,*args)
        # Save models/pars/chisq
        self._all_pars.append(list(args).copy())
        self._all_model.append(f0.copy())
        self._all_chisq.append(self.chisq(f0))
        chisq = np.sqrt( np.sum( (self._flux-f0)**2/self._err**2 )/len(self._flux) )

        # Loop over parameters
        #  stellar parameters and RVs for each spectrum
        parnames = self.fitparams.copy()+list('RV'+np.char.array((np.arange(nspec)+1).astype(str)))
        for i in range(npar):
            pars = np.array(copy.deepcopy(args))
            if i<nfitparams:
                step = self.getstep(self.fitparams[i])
            # RV
            else:
                step = 0.1
            # Check boundaries, if above upper boundary
            #   go the opposite way
            if pars[i]>ubounds[i]:
                step *= -1
            pars[i] += step
            
            if self.verbose:
                print('--- '+str(i+1)+' '+parnames[i]+' '+str(step)+' '+str(pars[i])+' ---')

            f1 = self.model(x,*pars)
            
            # Save models/pars/chisq
            self._all_pars.append(list(pars).copy())
            self._all_model.append(f1.copy())
            self._all_chisq.append(self.chisq(f1))
            
            if np.sum(~np.isfinite(f1))>0:
                print('some nans/infs')
                import pdb; pdb.set_trace()

            jac[:,i] = (f1-f0)/step
            
        if np.sum(~np.isfinite(jac))>0:
            print('some nans/infs')
            import pdb; pdb.set_trace()

        self._jac_array = jac.copy()   # keep a copy
            
        self.njac += 1
            
        return jac
