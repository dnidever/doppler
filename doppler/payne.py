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
from .spec1d import Spec1D
from . import utils
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
    files = glob(datadir+'*_NN_*.npz')
    nfiles = len(files)
    if nfiles==0:
        raise Exception("No Payne model files in "+datadir)
    coeffs, wavelength, labels = load_payne_model(files[0])
    return PayneModel(coeffs, wavelength, labels)


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
        labels = tmp["labels"]
    else:
        print('WARNING: No label array')
        labels = [None] * w_array_0.shape[1]
    coeffs = (w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max)
    tmp.close()
    return coeffs, wavelength, labels


class PayneModel(object):

    def __init__(self,coeffs,wavelength,labels):
        self._coeffs = coeffs
        self._dispersion = wavelength
        self.labels = labels

    #@property
    #def ranges(self):
    #    return self._data[0].ranges

    @property
    def dispersion(self):
        return self._dispersion
        
    def __call__(self,labels,wr=None,fluxonly=False):

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
            lo, = np.where(self.dispersion >= wr[0])
            hi, = np.where(self.dispersion <= wr[1])            
            if (len(lo)==0) | (len(hi)==0):
                raise Exception('No pixels between '+str(wr[0])+' and '+str(wr[1]))
            spectrum = spectrum[lo[0]:hi[0]+1]

        # Return as spectrum object with wavelengths
        if fluxonly is False:
            mspec = Spec1D(spectrum,wave=self._dispersion.copy(),lsfsigma=None,instrument='Model')
        else:
            mspec = spectrum
            
        return mspec


    def label_dict2array(self,labeldict):
        """ Convert labels from a dictionary or numpy structured array to array."""
        arr = np.zeros(len(self.labels),np.float64)
        for i in range(len(self.labels)):
            val = labeldict.get(self.labels[i])
            if val == None:
                raise ValueError(self.labels[i]+' NOT FOUND')
            arr[i] = val
        return arr
    
        
    def test(self,spec):
        # Fit a spectrum using this Payne Model
        # Outputs: labels, cov, meta
        
        # Flatten if multiple orders
        if self.norder>1:
            pmodel_flat = pmodel.flatten()
            flux = spec.flux.T.reshape(spec.npix*spec.norder)
            err = spec.err.T.reshape(spec.npix*spec.norder)
            cmodel = pmodel_flat._data[0]
            with mute():
                out = cmodel.test(flux, 1/err**2)
            return out
        else:
            flux = spec.flux
            err = spec.err
            cmodel = pmodel._data[0]
            with mute():
                out = cmodel.test(flux, 1/err**2)
            return out

    #def __setitem__(self,index,data):
    #    self._data[index] = data
    
    #def __getitem__(self,index):
    #    return self._data[index]
    

    def copy(self):
        # Make copies of the _data list of payne models
        new_coeffs = []
        for c in self._coeffs:
            new_coeffs.append(c.copy())
        new = PayneModel(new_coeffs,self._dispersion.copy(),self.labels.copy())
        return new
    
    def read(mfile):
        coeffs, wavelength, labels = load_payne_model(mfile)
        return PayneModel(coeffs, wavelength, labels)



class PayneModelSet(object):

    ## Set of Payne models that each cover a different "chunk" of wavelength
    
    def __init__(self,models):
        if type(models) is list:
            self.nmodel = len(models)
            self._data = models
            wr = np.zeros((len(models),2),np.float64)
            disp = []
            for i in range(len(models)):
                wr[i,0] = np.min(models[i].dispersion)
                wr[i,1] = np.max(models[i].dispersion)
                disp += list(models[i].dispersion)
            self._wr = wr
            self._dispersion = np.array(disp)
        else:
            self.nmodel = 1
            self._data = [models]    # make it a list so it is iterable
            wr = np.zeros(2,np.float64)
            wr[0] = np.min(models.dispersion)
            wr[1] = np.min(models.dispersion)            
            self._wr = wr
            self._dispersion = models._dispersion.copy()
        self.labels = models[0].labels

    #@property
    #def ranges(self):
    #    return self._data[0].ranges

    @property
    def dispersion(self):
        return self._dispersion
        
    def __call__(self,labels,wr=None):

        '''
        Predict the rest-frame spectrum (normalized) of a single star.
        We input the scaled stellar labels (not in the original unit).
        Each label ranges from -0.5 to 0.5
        '''

        # Only a subset of wavelenths requested
        if wr is not None:
            # Check that we have pixels in this range
            lo, = np.where(self._dispersion >= wr[0])
            hi, = np.where(self._dispersion <= wr[1])          
            if (len(lo)==0) | (len(hi)==0):
                raise Exception('No pixels between '+str(wr[0])+' and '+str(wr[1]))
            # Get the chunks that we need
            gg, = np.where( (self._wr[:,0] >= wr[0]) & (self._wr[:,1] <= wr[1]) )
            ngg = len(gg)
            npix = 0
            for i in range(ngg):
                npix += len(models[gg[i]].dispersion)
            spectrum = np.zeros(npix,np.float64)
            wave = np.zeros(npix,np.float64)
            cnt = 0
            for i in range(ngg):
                spec1 = models[gg[i]](labels,fluxonly=True)
                nspec1 = len(spec1)
                spectrum[cnt:cnt+nspec1] = spec1
                wave[cnt:cnt+nspec1] = models[gg[i]].dispersion               
                cnt += nspec1
            # Now trim a final time
            lo, = np.where(wave >= wr[0])
            hi, = np.where(wave <= wr[1])
            wave = wave[lo[0]:hi[0]+1]
            spectrum = spectrum[lo[0]:hi[0]+1]            
                               
            # Return as spectrum object with wavelengths
            mspec = Spec1D(spectrum,wave=wave,lsfsigma=None,instrument='Model')

        # all pixels
        else:
            spectrum = np.zeros(len(self._dispersion),np.float64)
            cnt = 0
            for i in range(self.nmodel):
                spec1 = self._data[i](labels,fluxonly=True)
                nspec1 = len(spec1)
                spectrum[cnt:cnt+nspec1] = spec1
                cnt += nspec1
                
            # Return as spectrum object with wavelengths
            mspec = Spec1D(spectrum,wave=self._dispersion.copy(),lsfsigma=None,instrument='Model')

        return mspec

    
    def test(self,spec):
        # Fit a spectrum using this Payne Model
        # Outputs: labels, cov, meta
        
        # Flatten if multiple orders
        if self.norder>1:
            pmodel_flat = pmodel.flatten()
            flux = spec.flux.T.reshape(spec.npix*spec.norder)
            err = spec.err.T.reshape(spec.npix*spec.norder)
            cmodel = pmodel_flat._data[0]
            with mute():
                out = cmodel.test(flux, 1/err**2)
            return out
        else:
            flux = spec.flux
            err = spec.err
            cmodel = pmodel._data[0]
            with mute():
                out = cmodel.test(flux, 1/err**2)
            return out

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

    
    
def prepare():
    """ Prepare a Payne spectrum for a given observed spectrum."""
    pass
