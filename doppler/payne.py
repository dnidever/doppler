#!/usr/bin/env python

"""PAYNE.PY - Routines to work with Payne models.

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20200815'  # yyyymmdd

# Most of the software is from Yuan-Sen Ting's The_Payne repository
# https://github.com/tingyuansen/The_Payne

import os
import numpy as np
import warnings
from glob import glob
from scipy.interpolate import interp1d
from dlnpyutils import (utils as dln, bindata, astro)
#from .spec1d import Spec1D
#from . import utils
from doppler.spec1d import Spec1D
from doppler import utils
import copy
import logging
import contextlib, io, sys

# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

cspeed = 2.99792458e5  # speed of light in km/s

# astropy.modeling can handle errors and constraints

    
def leaky_relu(z):
    '''
    This is the activation function used by default in all our neural networks.
    '''

    return z*(z > 0) + 0.01*z*(z < 0)
    


# Load the default Payne model
def load_model():
    """
    Load the default Payne model.
    """

    datadir = utils.datadir()
    files = glob(datadir+'payne_coolhot_*.npz')
    nfiles = len(files)
    if nfiles==0:
        raise Exception("No Payne model files in "+datadir)
    if nfiles>1:
        return PayneModelSet.read(files)
    else:
        return PayneModel.read(files)


# Load a single or list of Payne models
def load_payne_model(mfile):
    """
    Load a  Payne model from file.

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
        raise Exception("No Payne model files in "+datadir)
    return DopplerPayneModel.read(files)


def prepare_payne_model(model,labels,spec,rv=None,vmacro=None,vsini=None):
    """ Prepare a Payne spectrum for a given observed spectrum."""

    
    # Convert wavelength from air->vacuum or vice versa
    if model.wavevac != spec.wavevac:
        model.wavevac = spec.wavevac   # this will set things behind the scenes

    # Get full wavelength range and total wavelength coverage in the orders
    owr = dln.minmax(spec.wave)
    owavefull = dln.valrange(spec.wave)
    owavechunks = 0.0
    odw = np.zeros(spec.norder,np.float64)
    specwave = np.atleast_2d(spec.wave.copy()).T
    for o in range(spec.norder):
        owavechunks += dln.valrange(specwave[:,o])
        odw[o] = np.median(dln.slope(specwave[:,o]))
    # Get model spectrum for the entire wavelength range, across all orders
    # Or do them separately, if there are large gaps (e.g GALAH)
    if owavechunks > 0.5*owavefull:
        model_all_in_one = True
        nextend = int(np.ceil(spec.wave.size*0.10))  # extend 10% on each end
        w0 = np.maximum( owr[0]-nextend*np.median(odw), np.min(model.dispersion))
        w1 = np.minimum( owr[1]+nextend*np.median(odw), np.max(model.dispersion))
        modelspec_all = model(labels,wr=[w0,w1])
    else:
        model_all_in_one = False
        
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
        nextend = int(np.ceil(len(w)*0.25))  # extend 25% on each end
        nextend = np.maximum(nextend,200)    # or 200 pixels
        if model_all_in_one == True:
            if (np.min(model.dispersion)<(w0-nextend*meddw)) | (np.max(model.dispersion)>(w1+nextend*meddw)):            
                gd, = np.where( (modelspec_all.wave >= (w0-nextend*meddw)) |
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
        if nbin==0:
            raise Exception('Model has lower resolution than the observed spectrum')
        if nbin>1:
            npix2 = np.round(len(tmodelflux) // nbin).astype(int)
            rmodelflux = dln.rebin(tmodelflux[0:npix2*nbin],npix2)
            rmodelwave = dln.rebin(tmodelwave[0:npix2*nbin],npix2)
        else:
            rmodelflux = tmodelflux
            rmodelwave = tmodelwave
                
        # Convolve with LSF, Vsini and Vmacro kernels
        lsf = spec.lsf.anyarray(rmodelwave,xtype='Wave',order=o,original=False)
        cmodelflux = utils.convolve_sparse(rmodelflux,lsf)
        # Apply Vmacro broadening and Vsini broadening (km/s)
        if (vmacro is not None) | (vsini is not None):
            cmodelflux = utils.broaden(rmodelwave,cmodelflux,vgauss=vmacro,vsini=vsini)
            
        # Apply Radial Velocity
        if rv is not None:
            if rv != 0.0: rmodelwave *= (1+rv/cspeed)
        
        # Interpolate
        omodelflux = interp1d(rmodelwave,cmodelflux,kind='cubic',bounds_error=False,
                              fill_value=(np.nan,np.nan),assume_sorted=True)(w)
        
        # Order information
        omodel = Spec1D(omodelflux,wave=w)
        omodel.norder = norder
        omodel.order = o
                
        # Append to final output
        outmodel.append(omodel)

        
    # Single-element list
    if (type(outmodel) is list) & (len(outmodel)==1):
        outmodel = outmodel[0] 
            
    return outmodel



class PayneModel(object):

    def __init__(self,coeffs,wavelength,labels,wavevac=False):
        self._coeffs = coeffs
        self._dispersion = wavelength
        self.labels = list(labels)
        self._wavevac = wavevac
        wr = np.zeros(2,np.float64)
        wr[0] = np.min(wavelength)
        wr[1] = np.max(wavelength)
        self.wr = wr
        self.npix = len(self._dispersion)

    @property
    def dispersion(self):
        return self._dispersion

    @dispersion.setter
    def dispersion(self,disp):
        if len(disp) != len(self._dispersion):
            raise ValueError('Input dispersion array not of the right length')
        self._dispersion = disp
    
    @property
    def wavevac(self):
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


    def __call__(self,labels,spec=None,wr=None,rv=None,vsini=None,vmacro=None,fluxonly=False):

        if len(labels) != len(self.labels):
            raise ValueError('labels must have '+str(len(self.labels))+' elements')
        
        # Prepare the spectrum
        if spec is not None:
            out = self.prepare(labels,spec=spec,rv=rv,vsini=vsini,vmacro=vmacro)
            if fluxonly is True:
                return out.flux
            return out

        '''
        Predict the rest-frame spectrum (normalized) of a single star.
        We input the scaled stellar labels (not in the original unit).
        Each label ranges from -0.5 to 0.5
        '''
        
        # assuming your NN has two hidden layers.
        w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max = self._coeffs
        scaled_labels = (labels-x_min)/(x_max-x_min) - 0.5
        inside = np.einsum('ij,j->i', w_array_0, scaled_labels) + b_array_0
        outside = np.einsum('ij,j->i', w_array_1, leaky_relu(inside)) + b_array_1
        spectrum = np.einsum('ij,j->i', w_array_2, leaky_relu(outside)) + b_array_2

        # Trim
        if wr is not None:
            gd, = np.where( (self.dispersion >= wr[0]) & (self.dispersion <= wr[1]) )
            if len(gd)==0:
                raise Exception('No pixels between '+str(wr[0])+' and '+str(wr[1]))
            spectrum = spectrum[gd]


        # Apply Vmacro broadening and Vsini broadening (km/s)
        if (vmacro is not None) | (vsini is not None):
            spectrum = utils.broaden(self._dispersion,spectrum,vgauss=vmacro,vsini=vsini)
            
        # Return as spectrum object with wavelengths
        if fluxonly is False:
            mspec = Spec1D(spectrum,wave=self._dispersion.copy(),lsfsigma=None,instrument='Model')
        else:
            mspec = spectrum

        # Apply Radial Velocity
        if rv is not None:
            if rv != 0.0:
                wave = mspec.wave
                wave *= (1+rv/cspeed)
                mspec.wave = wave
            
        return mspec

    
    def label_arrayize(self,labeldict):
        """ Convert labels from a dictionary or numpy structured array to array."""
        arr = np.zeros(len(self.labels),np.float64)
        for i in range(len(self.labels)):
            val = labeldict.get(self.labels[i])
            if val == None:
                raise ValueError(self.labels[i]+' NOT FOUND')
            arr[i] = val
        return arr
    

    def copy(self):
        # Make copies of the _data list of payne models
        new_coeffs = []
        for c in self._coeffs:
            new_coeffs.append(c.copy())
        new = PayneModel(new_coeffs,self._dispersion.copy(),self.labels.copy())
        return new

    
    def read(mfile):
        """ Read in a single Payne Model."""
        coeffs, wavelength, labels, wavevac = load_payne_model(mfile)
        return PayneModel(coeffs, wavelength, labels, wavevac=wavevac)


    def prepare(self,labels,spec,rv=None,vmacro=None,vsini=None):
        return prepare_payne_model(self,labels,spec,rv=rv,vmacro=vmacro,vsini=vsini)


        
class PayneModelSet(object):

    ## Set of Payne models that each cover a different "chunk" of wavelength
    
    def __init__(self,models):
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
        
    @property
    def dispersion(self):
        return self._dispersion
    
    @property
    def wavevac(self):
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
    
    def __call__(self,labels,spec=None,wr=None,rv=None,vsini=None,vmacro=None,fluxonly=False):
        '''
        Predict the rest-frame spectrum (normalized) of a single star.
        We input the scaled stellar labels (not in the original unit).
        Each label ranges from -0.5 to 0.5
        '''

        if len(labels) != len(self.labels):
            raise ValueError('labels must have '+str(len(self.labels))+' elements')

        # Prepare the spectrum
        if spec is not None:
            out = self.prepare(labels,spec=spec,rv=rv,vsini=vsini,vmacro=vmacro)
            if fluxonly is True:
                return out.flux
            return out
            
        # Only a subset of wavelenths requested
        if wr is not None:
            # Check that we have pixels in this range
            lo, = np.where(self._dispersion >= wr[0])
            hi, = np.where(self._dispersion <= wr[1])          
            if (len(lo)==0) | (len(hi)==0):
                raise Exception('No pixels between '+str(wr[0])+' and '+str(wr[1]))
            # Get the chunks that we need
            gg, = np.where( (self._wrarray[0,:] >= wr[0]) & (self._wrarray[1,:] <= wr[1]) )
            ngg = len(gg)
            npix = 0
            for i in range(ngg):
                npix += self._data[gg[i]].npix
            spectrum = np.zeros(npix,np.float64)
            wave = np.zeros(npix,np.float64)
            cnt = 0
            for i in range(ngg):
                spec1 = self._data[gg[i]](labels,fluxonly=True)
                nspec1 = len(spec1)
                spectrum[cnt:cnt+nspec1] = spec1
                wave[cnt:cnt+nspec1] = self._data[gg[i]].dispersion               
                cnt += nspec1
            # Now trim a final time
            ggpix, = np.where( (wave >= wr[0]) & (wave <= wr[1]) )
            wave = wave[ggpix]
            spectrum = spectrum[ggpix]
                               
            # Return as spectrum object with wavelengths
            mspec = Spec1D(spectrum,wave=wave,lsfsigma=None,instrument='Model')

        # all pixels
        else:
            spectrum = np.zeros(self.npix,np.float64)
            cnt = 0
            for i in range(self.nmodel):
                spec1 = self._data[i](labels,fluxonly=True)
                nspec1 = len(spec1)
                spectrum[cnt:cnt+nspec1] = spec1
                cnt += nspec1
                
            # Return as spectrum object with wavelengths
            mspec = Spec1D(spectrum,wave=self._dispersion.copy(),lsfsigma=None,instrument='Model')

        # Apply Vmacro broadening and Vsini broadening (km/s)
        if (vmacro is not None) | (vsini is not None):
            oflux = utils.broaden(mspec.wave,mspec.flux,vgauss=vmacro,vsini=vsini)
            mspec.flux = oflux
            
        # Apply Radial Velocity
        if rv is not None:
            if rv != 0.0:
                wave = mspec.wave
                wave *= (1+rv/cspeed)
                mspec.wave = wave
            
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
    
    def read(mfiles):
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

    def prepare(self,labels,spec,rv=None,vmacro=None,vsini=None):
        return prepare_payne_model(self,labels,spec,rv=rv,vmacro=vmacro,vsini=vsini)


    
class DopplerPayneModel(object):
    """ Thin wrapper around PayneModel or PayneModelSet."""

    def __init__(self,model):
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
        
        
    # Maybe the __call__ inputs should be a PARAM dictionary, so it's
    # easier to specify what you want, and use "defaults" for the rest
    # of the parameters
        
    def __call__(self,labels,wr=None):
        """ Create the model."""
        # Dictionary input
        if isinstance(labels,dict):
            labels = self.mklabels(labels)  # convert dictionary to array of labels
            
        if len(labels) != len(self.labels):
            raise ValueError('labels must have '+str(len(self.labels))+' elements')
        vsini,vmacro,rv = labels[-3:]
        plabels = labels[0:-3]  # just the payne labels
        if self.prepared==True:
            return self._data(plabels,spec=self._spec,vsini=vsini,vmacro=vmacro,rv=rv)
        else:
            return self._data(plabels,vsini=vsini,vmacro=vmacro,rv=rv)        
        
    def mklabels(self,inputs):
        """ Convert input dictionary to labels."""

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
        return self._dispersion
        
    @property
    def wavevac(self):
        return self._data.wavevac
    
    @wavevac.setter
    def wavevac(self,wavevac):
        """ Set wavelength wavevac value."""
        self._data.wavevac = wavevac

    @property
    def prepared(self):
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

        
    def prepare(self,spec):
        """ Prepare the model.  Keep a copy of the spectrum."""
        self._spec = spec.copy()
        self._prepared = True
        self._dispersion = spec.wave.copy()
        self._data.wavevac = spec.wavevac   # switch air<->vac if necessary
        self._wavevac = spec.wavevac
        self.wr = dln.minmax(self._dispersion)

    def copy(self):
        """ Make a copy of the DopplerPayneModel."""
        new_model = self._data.copy()
        new = DopplerPayneModel(new_model)
        if self.prepared==True:
            new.prepare(self._spec)
        return new

    def read(mfiles):
        """ Read a set of Payne model files."""
        if dln.size(mfiles)==1:
            model = PayneModel.read(mfiles)
        else:
            model = PayneModelSet.read(mfiles)
        return DopplerPayneModel(model)
    


class PayneSpecFitter:

    def __init__(self,spec,pmodel,params,fitparams=None,verbose=False):
        # spec - observed spectrum object
        # pmodel - Payne model object
        # params - initial/fixed parameters dictionary
        # fitparams - parameter/label names to fit (default is all)
        # "Prepare" the Payne model with the observed spectrum
        pmodel.prepare(spec)
        self._paynemodel = pmodel
        self.labels = pmodel.labels
        labelnames = np.char.array(self._paynemodel.labels)        
        nlabels = len(self._paynemodel.labels)
        self.params = dict((key.upper(), value) for (key, value) in params.items()) # all CAPS
        self._initlabels = self.mkinitlabels(params)
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
                # Not in FITPARAMS but in PARAMS, FIXED                    
                elif name in self.params.keys():
                    fixed[k] = True
                # Not in FITPARAMS or PARAMS, but FE_H or ALPHA_H in FITPARAMS, NOT FIXED
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
                # Not in FITPARAMS but in PARAMS, FIXED
                elif name in self.params.keys():
                    fixed[k] = True
                # Not in FITPARAMS or PARAMS, but FE_H in FITPARAMS, NOT FIXED
                elif 'FE_H' in self.fitparams:
                    fixed[k] = False
                # Not in FITPARAMS/PARAMS and FE_H not being fit, FIXED
                else:
                    fixed[k] = True
            # Other parameters (Teff, logg, RV, Vturb, Vsini, etc.)
            else:
                # In FITPARAMS, NOT FIXED
                if name in self.fitparams:
                    fixed[k] = False
                # Not in FITPARAMS but in PARAMS, FIXED
                elif name in self.params.keys():
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
        self._verbse = verbose
        #self._norm = norm   # normalize
        self._continuum_func = spec.continuum_func
        # Figure out the wavelength parameters
        npix = spec.npix
        norder = spec.norder
        # parameters to save
        #self._all_pars = []
        #self._all_model = []
        #self._all_chisq = []
        #self._jac_array = None

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self,params):
        """ Dictionary, keys must be all CAPS."""
        self._params = dict((key.upper(), value) for (key, value) in params.items())  # all CAPS

    @property
    def fitparams(self):
        return self._fitparams

    @fitparams.setter
    def fitparams(self,fitparams):
        """ List, all CAPS."""
        self._fitparams = [f.upper() for f in fitparams]

    def mkinitlabels(self,inputs):
        """ Convert input dictionary to Payne labels."""

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
            # Only set the value if it was found in labelsnames
            labels[labelnames==name] = params[name]
            
        return labels

        
    def mklabels(self,args):
        """ Make labels for Payne model."""
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
                
        #import pdb; pdb.set_trace()

        
        #labels = self._initlabels.copy()
        ## Fitting ALPHA
        #if self._fitalpha is True:
        #    # Alpha element labels
        #    labels[self._fitalpha_index] = args[self._fitalpha_argindex]
        #    # Non-alpha element labels
        #    nonalpha_args = np.delete(args,self._fitalpha_argindex)
        #    labels[self._fitindex] = nonalpha_args
        ## Not fitting ALPHA
        #else:
        #    labels[self._fitindex] = args
        #return labels
    
    def chisq(self,model):
        return np.sqrt( np.sum( (self.flux-model)**2/self.err**2 )/len(self.flux) )
            
    def model(self,xx,*args):
        # Return model Payne spectrum given the input arguments."""
        # Convert arguments to Payne model inputs
        labels = self.mklabels(args)
        return self.paynemodel(labels)

    
