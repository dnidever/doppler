#!/usr/bin/env python

"""CANNON.PY - Routines to work with Cannon models.

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20210605'  # yyyymmdd

import os
import numpy as np
import warnings
from glob import glob
from scipy.interpolate import interp1d
import thecannon as tc
from thecannon.model import CannonModel
from thecannon import (censoring, vectorizer as vectorizer_module)
from dlnpyutils import (utils as dln, bindata, astro)
from .spec1d import Spec1D
from . import utils
import copy
import logging
import contextlib, io, sys
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

# Turn off the Cannon's info messages
tclogger = logging.getLogger('thecannon.utils')
tclogger.disabled = True

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


    
class DopplerCannonModelSet(object):
    """
    A class to wrap a set of Cannon models to be used by Doppler.
    Generally each Cannon model covers a different region of
    parameter space.

    Parameters
    ----------
    model : list of Cannon models
      List of Cannon models to use.

    """
    
    def __init__(self,models):
        """ Initialize DopplerCannonModelSet object."""
        if type(models) is list:
            self.nmodel = len(models)
            self._data = models
        else:
            self.nmodel = 1
            self._data = [models]    # make it a list so it is iterable
        self._prepared = False
        self.flattened = False
        self.labels = list(self._data[0].labels)
        self.wavevac = self._data[0].wavevac
        
    @property
    def has_continuum(self):
        """ Does the model have continuum parameters."""
        return self._data[0].has_continuum

    def get_best_model(self,pars):
        """ This returns the first DopplerCannonModel instance that has the right range."""
        for m in self._data:
            ranges = m.ranges
            inside = True
            for i in range(3):
                inside &= (pars[i]>=ranges[i,0]) & (pars[i]<=ranges[i,1])
            if inside:
                return m
        return None

    @property
    def dispersion(self):
        """ Wavelength array."""
        return self._data[0].dispersion
    
    def __call__(self,pars=None,teff=None,logg=None,feh=None,order=None,norm=True,
                 fluxonly=False,wave=None,rv=None):
        """
        Create the Cannon model spectrum given the input label values.  The appropriate
        Cannon model from the set is used.

        Parameters
        ----------
        pars : array, optional
            Array or list of Cannon parameters/labels.  This is normally [teff,logg,feh].
            If pars is not input then teff, logg, and feh need to be input separately.
        teff : float, optional
            Temperature value.  Used if pars is not input.
        logg : float, optional
            Surface gravity value.  Used if pars is not input.
        feh : float, optional
            Metallicity value.  Used if pars is not input.
        order : int, optional
            Which spectral order to use.  Default is to return all.
        norm : boolean, optional
            Return a normalized spectrum (norm=True, default), or a spectrum with
            the flux continuum included (norm=False).
        fluxonly : boolean, optional
            Only return the flux array.  Default is to return a Spec1D object.        
        wave : numpy array, optional
            Input wavelength array to use for the output Cannon model.  Default is to use the full
            wavelength range or the observed spectrum wavelengths if the model is "prepared".
        rv : float, optional
            Doppler shift to apply to the Cannon model (in km/s).  Default is no Doppler shift.

        Returns
        -------
        mspec : numpy array or Spec1D object
            The output model Cannon spectrum.  If fluxonly=True then only the flux array is returned,
            otherwise a Spec1D object is returned.

        Example
        -------
        .. code-block:: python

             mspec = model(labels)

        """
        
        # This will return a model given a set of parameters
        if pars is None:
            pars = np.array([teff,logg,feh])
            pars = pars.flatten()
        # Get best cannon model
        model = self.get_best_model(pars)
        if model is None: return None

        return model(pars,order=order,norm=norm,fluxonly=fluxonly,wave=wave,rv=rv)

    def model(self,spec,pars=None,teff=None,logg=None,feh=None,rv=None):
        """
        This will return a Cannon model given a set of parameters and prepared for an
        input observed spectrum.

        Parameters
        ----------
        spec : Spec1D object
          Observed spectrum to prepare the model with.
        pars : array, optional
            Array or list of Cannon parameters/labels.  This is normally [teff,logg,feh].
            If pars is not input then teff, logg, and feh need to be input separately.
        teff : float, optional
            Temperature value.  Used if pars is not input.
        logg : float, optional
            Surface gravity value.  Used if pars is not input.
        feh : float, optional
            Metallicity value.  Used if pars is not input.
        rv : float, optional
            Doppler shift to apply to the Cannon model (in km/s).  Default is no Doppler shift.

        Returns
        -------
        mspec : Spec1D object
            The output model Cannon spectrum.

        Example
        -------
        .. code-block:: python

             mspec = model.model(labels)

        """
        
        # This will return a model given a set of parameters, similar to model_spectrum()
        if rv is None: rv=0.0
        if pars is None:
            pars = np.array([teff,logg,feh])
            pars = pars.flatten()
        # Get best cannon model
        model = self.get_best_model(pars)
        if model is None: return None

        # NOT PREPARED, prepare for this spectrum and use its wavelength array
        if self.prepared is False:
            pmodel = self.prepare(spec)
        # Already PREPARED, just use the wavelength aray
        else:
            pmodel = self
            
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
    
            # Interpolation
            model_spec_interp = interp1d(model_wave1,model_spec1,kind='cubic',bounds_error=False,
                                         fill_value=(np.nan,np.nan),assume_sorted=True)(spwave1)
        
            # Stuff in the information for this order
            model_spec[:,o] = model_spec_interp
            model_wave[:,o] = spwave1
            sigma[:,o] = sigma1
        
        # Create Spec1D object
        mspec = Spec1D(model_spec,err=model_spec*0.0,wave=model_wave,lsfsigma=sigma,instrument='Model')
        for i in range(len(self.labels)):
            setattr(mspec,self.labels[i],labels[i])
        mspec.rv = rv
        #mspec.snr = np.inf
        
        return mspec

    def derivative(self,pars=None,teff=None,logg=None,feh=None,rv=None):
        """
        Return the derivative/Jacobian.

        Parameters
        ----------
        pars : array, optional
            Array or list of Cannon parameters/labels.  This is normally [teff,logg,feh].
            If pars is not input then teff, logg, and feh need to be input separately.
        teff : float, optional
            Temperature value.  Used if pars is not input.
        logg : float, optional
            Surface gravity value.  Used if pars is not input.
        feh : float, optional
            Metallicity value.  Used if pars is not input.
        rv : float, optional
            Doppler shift to apply to the Cannon model (in km/s).  Default is no Doppler shift.

        Returns
        -------
        deriv : numpy array
            The derivative of the spectrum with respect to the labels [Nlabels,Npix,Norder].
              If rv=None, then the RV dimension is not included.  If Norder=1, then the
              "extra" last dimension is trimmed.

        Example
        -------
        .. code-block:: python

             der = derivative(pars)

        """
        
        if pars is None:
            pars = np.array([teff,logg,feh])
            pars = pars.flatten()
        # Get best cannon model
        model = self.get_best_model(pars)
        if model is None: return None
        return model.derivative(pars,rv=rv)
    
    def test(self,spec):
        """
        Fit the Cannon model label values for a given input spectrum.

        Parameters
        ----------
        spec : Spec1D object
           The observed spectrum to fit.

        Results
        -------
        out : list
           List of best-fit label values.

        Example
        -------
        .. code-block:: python

             labels = model.test(spec)        

        """
        # fit a spectrum across all the models, return the best one

        if spec.normalized is False:
            print('WARNING: Spectrum is NOT normalized')
        
        # Loop over all models
        out = []
        bestmodel = []
        chisq = []
        for m in self:
            with mute():
                labels1, cov1, meta1 = m.test(spec)
            # Take out the first dimension
            labels1 = labels1.flatten()
            cov1 = cov1[0,:,:]
            out1 = (labels1,cov1,meta1)
            out.append(out1)
            if np.isfinite(labels1[0]):
                bestmodel1 = m(labels1)
                bestmodel.append(bestmodel1)
                chisq1 = np.sqrt(np.sum(((spec.flux-bestmodel1.flux)/spec.err)**2))
                chisq.append(chisq1)
            else:
                bestmodel.append(np.nan)
                chisq.append(1e50)
        # Find the best one
        chisq = np.array(chisq)
        bestind = np.argmin(np.array(chisq))
        return out[bestind]
        
    def __len__(self):
        return self.nmodel

    def __setitem__(self,index,data):
        self._data[index] = data
    
    def __getitem__(self,index):
        return self._data[index]
    
    def __iter__(self):
        self._count = 0
        return self
        
    def __next__(self):
        if self._count < self.nmodel:
            self._count += 1            
            return self._data[self._count-1]
        else:
            raise StopIteration

    def prepare(self,spec):
        """
        Prepare the model using an observed spectrum.  This keeps a copy of the spectrum,
        makes sure the model wavevac matches the observed spectrum value, and convolves
        the Cannon model with the observed spectrum's LSF.

        After a DopplerCannonModelSet has been prepared, then using __call__ will automatically
        return a Cannon model spectrum that has been convolved with the observed spectrum's
        LSF and on its wavelength array.

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
        
        # Loop through the models and prepare them
        self._original_model = self.copy()
        for i,m in enumerate(self):
            m.prepare(spec)
        self._prepared = True
        self.wavevac = self._data[0].wavevac

    @property
    def prepared(self):
        """ Has the model been prepared with an observd spectrum."""
        return self._prepared

    @prepared.setter
    def prepared(self,value):
        """ Can be used to unprepare() the model."""
        if (value==0) | (value==False):
            self.unprepare()
        
    def unprepare(self):
        """ Set back to original."""
        # Copy the original model back
        self.nmodel = self._original_model.nmodel
        self._data = self._original_model._data.copy()
        self.wavevac = self._original_model.wavevac
        self._original_model = None
        self._prepared = False
        self.flattened = False
            
    def interp(self,wave):
        """ Interpolate model onto a new wavelength grid. """
        # Interpolate onto a new wavelength scale
        models = []
        for i,m in enumerate(self):
            newm = m.interp(wave)
            models.append(newm)
        new = DopplerCannonModelSet(models)
        return new
    
    def flatten(self):
        """ Returns the model with multiple orders flattened and stacked into one long model """
        # This flattens multiple orders and stacks them into one long model
        if self[0].norder==1:
            return self.copy()
        models = []
        for i,m in enumerate(self._data):
            models.append(m.flatten())
        new = DopplerCannonModelSet(model)
        return new

    def copy(self):
        """ Make a copy of the DopplerCannonModel."""
        # Make a copy of this DopplerCannonModelSet
        models = []
        for i,m in enumerate(self):
            models.append(m.copy())
        new = DopplerCannonModelSet(models)
        return new

    @classmethod
    def read(cls,files):
        """ Read a list of Cannon model files."""
        models = load_cannon_model(files)
        return DopplerCannonModelSet(models)
    

    
class DopplerCannonModel(object):
    """
    A class to wrap a Cannon model to be used by Doppler.

    Parameters
    ----------
    model : Cannon model
      The Cannon model to use.

    """
    
    def __init__(self,model):
        """ Initialize DopplerCannonModel object."""
        if type(model) is list:
            if len(model)>1:
                raise ValueError("Can only input a single Cannon model")
        self.norder = 1
        if type(model) is list:
            self._data = model
        else:
            self._data = [model]     # make it a list so it is iterable
        self._prepared = False
        self.flattened = False
        self.labels = list(self._data[0].vectorizer.label_names)
        self.wavevac = self._data[0].wavevac
        
    @property
    def has_continuum(self):
        """ Does the model have continuum parameters."""
        return hasattr(self._data[0],'continuum')

    @property
    def ranges(self):
        """ Parameter ranges."""
        return self._data[0].ranges

    @property
    def dispersion(self,order=0):
        """ Wavelength array."""
        return self._data[order].dispersion
        
    def __call__(self,labels,order=None,norm=True,fluxonly=False,wave=None,rv=None):
        """
        Create the Cannon model spectrum given the input label values.

        Parameters
        ----------
        labels : list or array
            List/array or dictionary of input labels values to use.
        order : int, optional
            Which spectral order to use.  Default is to return all.
        norm : boolean, optional
            Return a normalized spectrum (norm=True, default), or a spectrum with
            the flux continuum included (norm=False).
        fluxonly : boolean, optional
            Only return the flux array.  Default is to return a Spec1D object.        
        wave : numpy array, optional
            Input wavelength array to use for the output Cannon model.  Default is to use the full
            wavelength range or the observed spectrum wavelengths if the model is "prepared".
        rv : float, optional
            Doppler shift to apply to the Cannon model (in km/s).  Default is no Doppler shift.

        Returns
        -------
        mspec : numpy array or Spec1D object
            The output model Cannon spectrum.  If fluxonly=True then only the flux array is returned,
            otherwise a Spec1D object is returned.

        Example
        -------
        .. code-block:: python

             mspec = model(labels)

        """
        
        # Default is to return all orders
        # order can also be a list or array of orders
        # Orders to loop over
        if order is None:
            orders = np.arange(self.norder)
        else:
            orders = list(np.atleast_1d(order))
        norders = dln.size(orders)
        # Get maximum number of pixels over all orders
        npix = 0
        for i in range(self.norder):
            npix = np.maximum(npix,self._data[i].dispersion.shape[0])
        # Wavelength array input
        if wave is not None:
            if wave.ndim==1:
                wnorders = 1
            else:
                wnorders = wave.shape[1]
            if wnorders != norders:
                raise ValueError("Number of orders in WAVE must match orders in the model")
            npix = wave.shape[0]
        # Use original input spectrum wavelength array
        elif wave is None and self.prepared:
            if hasattr(self,'_specwave')==False:
                import pdb; pdb.set_trace()
            npix = self._specwave.shape[0]
        # Initialize output arrays
        oflux = np.zeros((npix,norders),np.float32)
        owave = np.zeros((npix,norders),np.float64)
        omask = np.zeros((npix,norders),bool)+True
        # Order loop
        for i in orders:
            if wave is None:
                # Use original input wavelengths
                if self.prepared:
                    if self._specwave.ndim==1:
                        owave1 = self._specwave.copy()
                    else:
                        owave1 = self._specwave[:,i].copy()
                # User cannon model wavelengths
                else:
                    owave1 = self._data[i].dispersion  # final wavelength array for this order
            else:
                if wave.ndim==1:
                    owave1 = wave.copy()
                else:
                    owave1 = wave[:,i]
            # Get model and add radial velocity if necessary
            if (rv is None) & (wave is None):
                m = self._data[i]
                f = m(labels)
                zfactor = 1
            else:
                if self.prepared:
                    m = self._data_nointerp[i]
                else:
                    m = self._data[i]
                f0 = m(labels)
                zfactor = 1 + rv/cspeed   # redshift factor
                zwave = m.dispersion*zfactor  # redshift the wavelengths
                f = np.zeros(len(owave1),np.float32)
                gind,ngind = dln.where((owave1>=np.min(zwave)) & (owave1<=np.max(zwave)))  # wavelengths we can cover
                if ngind>0:
                    f[gind] = dln.interp(zwave,f0,owave1[gind])
            # Force normalized flux <=1.0
            #  sometimes there are "spikes" above 1.0
            f = np.minimum(f,1.0)
            # Get Continuum
            if (norm is False):
                if hasattr(m,'continuum'):
                    contmodel = m.continuum
                    smallcont = contmodel(labels)
                    if contmodel._logflux is True:
                        smallcont = 10**smallcont
                    # Interpolate to the full spectrum wavelength array
                    #   with any redshift
                    cont = dln.interp(contmodel.dispersion*zfactor,smallcont,owave1)
                    # Now mulitply the flux array by the continuum
                    f *= cont
                else:
                    raise ValueError("Model does not have continuum information")
            # Stuff in the array
            oflux[0:len(f),i] = f
            owave[0:len(f),i] = owave1
            omask[0:len(f),i] = False
            
        # Change single order 2D arrays to 1D
        if norders==1:
            oflux = oflux.flatten()
            owave = owave.flatten()
            omask = omask.flatten()            
            
        # Only return the flux
        if fluxonly is True: return oflux
        
        # Create Spec1D object
        mspec = Spec1D(oflux,err=oflux*0.0,wave=owave,mask=omask,lsfsigma=None,instrument='Model')
        for i in range(len(self.labels)):
            setattr(mspec,self.labels[i],labels[i])
        #mspec.snr = np.inf
        
        return mspec

    def derivative(self,labels,rv=None):
        """
        Return the derivative/Jacobian.

        Parameters
        ----------
        labels : list or array
            List/array or dictionary of input labels values to use.
        rv : float, optional
            Doppler shift to apply to the Cannon model (in km/s).  Default is no Doppler shift.

        Returns
        -------
        deriv : numpy array
            The derivative of the spectrum with respect to the labels [Nlabels,Npix,Norder].
              If rv=None, then the RV dimension is not included.  If Norder=1, then the
              "extra" last dimension is trimmed.

        Example
        -------
        .. code-block:: python

             der = derivative(labels)

        """
            
        # Loop over orders
        if rv is None:
            deriv = np.zeros((3,len(self.dispersion),self.norder),float)
        else:
            deriv = np.zeros((4,len(self.dispersion),self.norder),float)            
        for o in range(self.norder):
            # NO RVs to deal with
            if (rv is None) or (self.prepared==False):
                model1 = self._data[o]
            # Need to deal with RV and doppler shift
            else:
                model1 = self._data_nointerp[o]
            # Use the label vector derivative.
            scaled_labels = (np.atleast_2d(labels) - model1._fiducials)/model1._scales
            scaled_labels = scaled_labels[0,:]
            dflux = np.dot(model1.theta,model1.vectorizer.get_label_vector_derivative(scaled_labels)).T
            # This is the derivative w.r.t the scaled labels
            #  need to use _scales to scale it
            dflux /= model1._scales.reshape(-1,1)

            # Deal with doppler shift if necessary
            if (rv is not None):
                owave1 = self._data[o].dispersion
                zfactor = 1 + rv/cspeed   # redshift factor
                zwave = model1.dispersion*zfactor  # redshift the wavelengths
                gind,ngind = dln.where((owave1>=np.min(zwave)) & (owave1<=np.max(zwave)))  # wavelengths we can cover
                if ngind>0:
                    # Loop over labels
                    for i in range(3):
                        deriv[i,gind,o] = dln.interp(zwave,dflux[i,:],owave1[gind])                        
            # No doppler shift to deal with
            else:
                deriv[:,:,o] = dflux

        # Add derivative for RV
        if rv is not None:
            flux0 = self(labels,rv=rv,fluxonly=True)
            flux1 = self(labels,rv=rv+1,fluxonly=True)
            if self.norder==1:
                deriv[3,:,0] = flux1.squeeze()-flux0.squeeze()
            else:
                deriv[3,:,:] = flux1-flux0                
                
        if self.norder==1:
            deriv = deriv.squeeze()
            
        return deriv
    
    
    def test(self,spec):
        """
        Fit the Cannon model label values for a given input spectrum.

        Parameters
        ----------
        spec : Spec1D object
           The observed spectrum to fit.

        Results
        -------
        out : list
           List of best-fit label values.

        Example
        -------
        .. code-block:: python

             labels = model.test(spec)        

        """
        # Fit a spectrum using this Cannon Model
        # Outputs: labels, cov, meta
        
        # NOT PREPARED, prepare for this spectrum and use its wavelength array
        if self.prepared is False:
            pmodel = self.prepare(spec)
        # Already PREPARED
        else:
            pmodel = self
            
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

    def __len__(self):
        return self.norder

    def __setitem__(self,index,data):
        self._data[index] = data
    
    def __getitem__(self,index):
        return self._data[index]
    
    def prepare(self,spec):
        """
        Prepare the model using an observed spectrum.  This keeps a copy of the spectrum,
        makes sure the model wavevac matches the observed spectrum value, and convolves
        the Cannon model with the observed spectrum's LSF.

        After a DopplerCannonModel has been prepared, then using __call__ will automatically
        return a Cannon model spectrum that has been convolved with the observed spectrum's
        LSF and on its wavelength array.

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
        
        # prepare the models
        # already prepared, unprepare, then prepare again
        if self.prepare is True:
            self.unprepare()
            self.prepare(spec)
            return
        # Keep a copy of the original models
        self._original_model = self.copy()
        # Keep uninterpolated model in ._data_nointerp, and the interpolated
        #  data in ._data
        #  a model for each order
        newm_nointerp = prepare_cannon_model(self._data[0],spec,dointerp=False)
        if type(newm_nointerp) is not list: newm_nointerp=[newm_nointerp]  # make sure it's a list
        # Interpolate onto the observed wavelength scale
        newm = []
        for i in range(spec.norder):
            wave = spec.wave
            if wave.ndim==1: wave=np.atleast_2d(spec.wave).T
            gdpix, = np.where((wave[:,i]>0) & np.isfinite(wave[:,i]))
            newm1 = interp_cannon_model(newm_nointerp[i],wout=wave[gdpix,i])
            newm1.interp = True
            newm.append(newm1)

        self._specwave = spec.wave.copy()  # save a copy of the original wavelengths
        self._data = newm
        self._data_nointerp = newm_nointerp
        self.norder = len(newm)
        self._prepared = True
        self.wavevac = self._data[0].wavevac
        
    @property
    def prepared(self):
        """ Has the model been prepared with an observd spectrum."""
        return self._prepared

    @prepared.setter
    def prepared(self,value):
        """ Can be used to unprepare() the model."""
        if (value==0) | (value==False):
            self.unprepare()
        
    def unprepare(self):
        """ Set back to original model."""
        # Copy the original model back
        self._data = self._original_model._data.copy()
        self._data_nointerp = None
        self.wavevac = self._original_model.wavevac
        self._original_model = None
        self._prepared = False
        self._specwave = None
        self.flattened = False
    
    def interp(self,wave):
        """ Interpolate model onto a new wavelength grid. """
        # Interpolate model onto new wavelength grid
        if wave.ndim==1:
            wnorder = 1
            wave = np.atleast_2d(wave).T
        else:
            wnorder = wave.shape[1]
        if wnorder != self.norder:
            raise ValueError('Wave must have same orders as cannon model')
        newm = []
        for i in range(self.norder):
            newm1 = interp_cannon_model(self._data_nointerp[i],wout=wave[:,i])
            newm1.interp = True
            newm.append(newm1)
        new = DopplerCannonModel(newm[0])
        new._data = newm
        newm_nointerp = []
        for i in range(self.norder):
            newm_nointerp.append(cannon_copy(self._data_nointerp[i]))
        new._data_nointerp = newm_nointerp
        new._prepared = self._prepared
        new._specwave = wave.copy()   # Save the input wavelengths
        new.norder = len(newm)
        return new
            
    def flatten(self):
        """ Returns the model with multiple orders flattened and stacked into one long model """
        # This flattens multiple orders and stacks then into one long model
        # THIS REMOVES THE _DATA_NOINTERP INFORMATION
        if self.norder==1:
            return self.copy()
        # stack them, spectrum and continuum
        new_data = hstack(self._data)
        new = DopplerCannonModel(new_data)
        # data_nointerp
        if self.prepared is True:
            new_data_nointerp = hstack(self._data_nointerp)
            new._data_nointerp = new_data_nointerp
        new._prepared = self._prepared
        new.flattened = True
        return new

    def copy(self):
        """ Make a copy of the DopplerCannonModel."""
        # Make copies of the _data list of cannon models
        new_data = []
        for d in self._data:
            new_data.append(cannon_copy(d))
        new = DopplerCannonModel(new_data[0])
        new._data = new_data
        new.norder = len(new_data)
        if (self.prepared is True) & (self.flattened is False):
            new_data_nointerp = []
            for d in self._data_nointerp:
                new_data_nointerp.append(cannon_copy(d))
            new._data_nointerp = new_data_nointerp
        new._prepared = self._prepared
        new.flattened = self.flattened
        # Copy _original_model and _specwave
        if hasattr(self,'_specwave'):
            new._specwave = self._specwave.copy()
        if hasattr(self,'_original_model'):
            new._original_model = self._original_model.copy()
        return new
    
    @classmethod
    def read(cls,mfile):
        """ Read a Cannon model file."""
        model = load_cannon_model(mfile)
        return DopplerCannonModel(model)

    
    
def cannon_copy(model):
    """ Make a new copy of a Cannon model."""
    npix, ntheta = model._theta.shape
    nlabels = len(model.vectorizer.label_names)
    labelled_set = np.zeros([2,nlabels])
    normalized_flux = np.zeros([2,npix])
    normalized_ivar = normalized_flux.copy()*0
    # Vectorizer
    vclass = type(model.vectorizer)
    vec = vclass(label_names=copy.deepcopy(model.vectorizer._label_names),
                                             terms=copy.deepcopy(model.vectorizer._terms))
    # Censors
    censors = censoring.Censors(label_names=copy.deepcopy(model.censors._label_names),
                                num_pixels=copy.deepcopy(model.censors._num_pixels))
    # Make new cannon model
    omodel = tc.CannonModel(labelled_set,normalized_flux,normalized_ivar,vectorizer=vec,
                            dispersion=copy.deepcopy(model.dispersion),regularization=copy.deepcopy(model.regularization),
                            censors=censors)        
    # Copy over all of the attributes
    for name, value in vars(model).items():
        if name not in ['_vectorizer','_censors','_regularization','_dispersion','continuum']:
            setattr(omodel,name,copy.deepcopy(value))
    # Continuum
    if hasattr(model,'continuum'):
        omodel.continuum = cannon_copy(model.continuum)

    return omodel


def readfromdata(state):
    """ Read a Cannon model from the loaded pickle data/dictionary.
        This is used to generate the continuum model."""
    
    metadata = state.get("metadata", {})
    
    init_attributes = list(metadata["data_attributes"]) \
                      + list(metadata["descriptive_attributes"])

    kwds = dict([(a, state.get(a, None)) for a in init_attributes])
        
    # Initiate the vectorizer.
    vectorizer_class, vectorizer_kwds = kwds["vectorizer"]
    klass = getattr(vectorizer_module, vectorizer_class)
    kwds["vectorizer"] = klass(**vectorizer_kwds)
    
    # Initiate the censors.
    kwds["censors"] = censoring.Censors(**kwds["censors"])

    model = CannonModel(**kwds)

    # Set training attributes.
    for attr in metadata["trained_attributes"]:
        setattr(model, "_{}".format(attr), state.get(attr, None))

    return model


def hstack(models):
    """
    Stack Cannon models.  Basically combine all of the pixels right next to each other.

    Parameters
    ----------
    models : list of Cannon models
       List of Cannon models to stack

    Returns
    -------
    omodel : Cannon model
       Output Cannon model with input Cannon pixel values next to each other.

    Example
    -------
    .. code-block:: python
             
         omodel = hstack(models)

    """
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

    # Vectorizer
    vclass = type(models[0].vectorizer)
    vec = vclass(label_names=copy.deepcopy(models[0].vectorizer._label_names),
                 terms=copy.deepcopy(models[0].vectorizer._terms))
    # Censors
    censors = censoring.Censors(label_names=copy.deepcopy(models[0].censors._label_names),
                                num_pixels=copy.deepcopy(models[0].censors._num_pixels))
    # Make new cannon model
    omodel = tc.CannonModel(labelled_set,normalized_flux,normalized_ivar,vectorizer=vec,
                            regularization=copy.deepcopy(models[0].regularization),
                            censors=censors)        
    omodel._s2 = np.zeros(nfpix,np.float64)
    omodel._scales = models[0]._scales.copy()
    omodel._theta = np.zeros((nfpix,ntheta),np.float64)
    omodel._design_matrix = models[0]._design_matrix.copy()
    omodel._fiducials = models[0]._fiducials.copy()
    omodel.dispersion = np.zeros(nfpix,np.float64)
    omodel.regularization = models[0].regularization
    if hasattr(models[0],'ranges'):
        omodel.ranges = models[0].ranges

    # scales, design_matrix, fiducials should be identical or we have problems
    if hasattr(models[0],'ranges'):
        if (np.sum((models[0]._scales!=models[1]._scales)) + np.sum((models[0]._design_matrix!=models[1]._design_matrix)) + \
            np.sum((models[0]._fiducials!=models[1]._fiducials)) + np.sum((models[0].ranges!=models[1].ranges))) > 0:
            raise ValueError('scales, design_matrix, fiducials, and ranges must be identical in the Cannon models')
    else:
        if (np.sum((models[0]._scales!=models[1]._scales)) + np.sum((models[0]._design_matrix!=models[1]._design_matrix)) + \
            np.sum((models[0]._fiducials!=models[1]._fiducials))) > 0:
            raise ValueError('scales, design_matrix, and fiducials must be identical in the Cannon models')
        
    # Fill in the information
    off = 0  # offset
    for i in range(nmodels):
        model = models[i]
        npix, ntheta = model._theta.shape
        omodel._s2[off:off+npix] = model._s2
        omodel._theta[off:off+npix,:] = model._theta
        omodel.dispersion[off:off+npix] = model.dispersion
        off += npix

    # Stack the continuum models as well
    if hasattr(model,'continuum'):
        cmodels = []
        for i in range(nmodels):
            cmodels.append(models[i].continuum)
        contstack = hstack(cmodels)
        omodel.continuum = contstack

    if hasattr(model,'fwhm') is True: omodel.fwhm=models[0].fwhm
    if hasattr(model,'wavevac') is True: omodel.wavevac=models[0].wavevac
        
    return omodel


# Load a single or list of Cannon models
def load_cannon_model(files):
    """
    Load a single (or list of) Cannon models from file and manipulate as needed.

    Parameters
    ----------
    files : list
       List of Cannon model files to load.

    Returns
    -------
    files : string
       File name (or list of filenames) of Cannon models to load.

    Example
    -------
    .. code-block:: python

         model = load_cannon_model(files)

    """

    # Convert single string into a list
    if type(files) is not list: files=list(np.atleast_1d(files))

    model = []
    for f in files:
        if os.path.exists(f) == False:
            raise ValueError(f+' not found')
        model1 = tc.CannonModel.read(f)
        # Generate the model ranges
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
        # Create continuum model if the information is there
        if hasattr(model1,'_continuum'):
            contmodel = readfromdata(model1._continuum)
            model1.continuum = contmodel
            delattr(model1,'_continuum')
        # Convert to DopplerCannonModel
        model.append(DopplerCannonModel(model1))

    if len(model)==1: model = model[0]
    return model



# Load all cannon models and return as DopplerCannonModelSet.
def load_models():
    """
    Load all Cannon models from the Doppler data/ directory
    and return as a DopplerCannonModelSet.

    Returns
    -------
    models : DopplerCannonModelSet
        DopplerCannonModelSet for all cannon models in the
        Doppler /data directory.

    Examples
    --------
    .. code-block:: python

         models = load_models()

    """    
    datadir = utils.datadir()
    files = glob(datadir+'cannongrid*.pkl')
    nfiles = len(files)
    if nfiles==0:
        # No Cannon model files.  Download them
        print("No Cannon model files in "+datadir+". Downloading them.")
        utils.download_data()
        #raise Exception("No Cannon model files in "+datadir)
    models = load_cannon_model(files)
    return DopplerCannonModelSet(models)


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
    .. code-block:: python

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

    Example
    -------
    .. code-block:: python

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
        if hasattr(model,'fwhm') is True: omodel.fwhm=model.fwhm
        if hasattr(model,'wavevac') is True: omodel.wavevac=model.wavevac
        
        # Copy continuum information
        if hasattr(model,'continuum'):
            omodel.continuum = cannon_copy(model.continuum)

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

    Example
    -------
    .. code-block:: python

         omodel = rebin_cannon_model(model,4)

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
        if hasattr(model,'fwhm') is True: omodel.fwhm=model.fwhm        
        if hasattr(model,'wavevac') is True: omodel.wavevac=model.wavevac
        
        # Copy continuum information
        if hasattr(model,'continuum'):
            omodel.continuum = cannon_copy(model.continuum)
        
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
      The desired output wavelength array.

    Returns
    -------
    omodel : Cannon model or list
      The interpolated Cannon model or list of models.

    Example
    -------
    .. code-block:: python

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
                            fill_value='extrapolate',assume_sorted=True)(wout)
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
                                         fill_value='extrapolate',assume_sorted=True)(xout)
        omodel.regularization = model.regularization
        if hasattr(model,'ranges') is True: omodel.ranges=model.ranges
        if hasattr(model,'fwhm') is True: omodel.fwhm=model.fwhm        
        if hasattr(model,'wavevac') is True: omodel.wavevac=model.wavevac
        
        # Copy continuum information
        if hasattr(model,'continuum'):
            omodel.continuum = cannon_copy(model.continuum)
        
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

    Example
    -------
    .. code-block:: python

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
        if hasattr(model,'fwhm') is True: omodel.fwhm=model.fwhm        
        if hasattr(model,'wavevac') is True: omodel.wavevac=model.wavevac
        
        # Copy continuum information
        if hasattr(model,'continuum'):
            omodel.continuum = cannon_copy(model.continuum)
        
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
       The default is False.

    Returns
    -------
    omodel : Cannon model or list
        The Cannon model or list of models prepared for "spec".

    Example
    -------
    .. code-block:: python

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

        # Convert wavelength from air->vacuum or vice versa
        if model.wavevac != spec.wavevac:
            # Air -> Vacuum
            if spec.wavevac is True:
                model.dispersion = astro.airtovac(model.dispersion)
                model.wavevac = True
            # Vacuum -> Air
            else:
                model.dispersion = astro.vactoair(model.dispersion)
                model.wavevac = False

        # Observed spectrum values
        wave = spec.wave
        mask = spec.mask
        ndim = wave.ndim
        if ndim==2:
            npix,norder = wave.shape
        if ndim==1:
            norder = 1
            npix = len(wave)
            wave = wave.reshape(npix,norder)
            mask = mask.reshape(npix,norder)
        # Loop over the orders
        outmodel = []
        for o in range(norder):
            m = mask[:,o]
            w = wave[:,o]
            gdpix, = np.where(~m)
            if len(gdpix)==0:
                raise ValueError('No unmasked pixels in order ',o)
            w = w[gdpix]
            w0 = np.min(w)
            w1 = np.max(w)
            dw = np.gradient(w)
            dw = np.abs(dw)
            meddw = np.median(dw)
            npix = len(w)

            if (np.min(model.dispersion)>w0) | (np.max(model.dispersion)<w1):
                raise Exception('Model does not cover the observed wavelength range')
            
            # Trim
            nextend = int(np.ceil(npix*0.25))  # extend 25% on each end
            nextend = np.maximum(nextend,200)    # or 200 pixels
            if (np.min(model.dispersion)<(w0-nextend*meddw)) | (np.max(model.dispersion)>(w1+nextend*meddw)):
                tmodel = trim_cannon_model(model,w0=w0-nextend*meddw,w1=w1+nextend*meddw)
            #if (np.min(model.dispersion)<(w0-50*meddw)) | (np.max(model.dispersion)>(w1+50*meddw)):
            #    tmodel = trim_cannon_model(model,w0=w0-20*meddw,w1=w1+20*meddw)
            #if (np.min(model.dispersion)<(w0-1000.0*w0/cspeed)) | (np.max(model.dispersion)>(w1+1000.0*w1/cspeed)):                                
            #    # allow for up to 1000 km/s offset on each end
            #    tmodel = trim_cannon_model(model,w0=w0-1000.0*w0/cspeed,w1=w1+1000.0*w1/cspeed)
                tmodel.trim = True
            else:
                tmodel = cannon_copy(model)
                tmodel.trim = False
                
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
            # need at least ~4 pixels per LSF FWHM across the spectrum
            #  using 3 affects the final profile shape
            nbin = np.round(np.min(fwhmpix)//4).astype(int)
            if np.min(fwhmpix) < 3.7:
                warnings.warn('Model has lower resolution than the observed spectrum. Only '+str(np.min(fwhmpix))+' model pixels per resolution element')
            if np.min(fwhmpix) < 2.8:
                raise Exception('Model has lower resolution than the observed spectrum. Only '+str(np.min(fwhmpix))+' model pixels per resolution element')
            
            if nbin>1:
                rmodel = rebin_cannon_model(tmodel,nbin)
                rmodel.rebin = True
            else:
                rmodel = cannon_copy(tmodel)
                rmodel.rebin = False
                
            # Convolve
            lsf = spec.lsf.anyarray(rmodel.dispersion,xtype='Wave',order=o,original=False)
            cmodel = convolve_cannon_model(rmodel,lsf)
            cmodel.convolve = True
            
            # Interpolate
            if dointerp is True:
                omodel = interp_cannon_model(cmodel,wout=w)
                omodel.interp = True
            else:
                omodel = cannon_copy(cmodel)
                omodel.interp = False
                
            # Order information
            omodel.norder = norder
            omodel.order = o

          #  # Copy continuum information
          #  if hasattr(model,'continuum'):
          #      omodel.continuum = cannon_copy(model.continuum)
          
            # Append to final output
            outmodel.append(omodel)

    # Single-element list
    if (type(outmodel) is list) & (len(outmodel)==1):
        outmodel = outmodel[0] 
        
    return outmodel


def model_spectrum(models,spec,teff=None,logg=None,feh=None,rv=None):
    """
    Create a Cannon model spectrum matched to an input observed spectrum
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

    Example
    -------
    .. code-block:: python

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
    mspec = Spec1D(model_spec,err=model_spec*0.0,wave=model_wave,lsfsigma=sigma,instrument='Model')
    mspec.teff = teff
    mspec.logg = logg
    mspec.feh = feh
    mspec.rv = rv
    #mspec.snr = np.inf
        
    return mspec

