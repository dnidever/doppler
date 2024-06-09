#!/usr/bin/env python

"""SPEC1D.PY - The spectrum 1D class and methods

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20210605'  # yyyymmdd                                                                                                                           

import time
import numpy as np
import math
import warnings
from scipy.interpolate import interp1d
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.io import fits
from astropy.table import Table
import thecannon as tc
from dlnpyutils import utils as dln, bindata, astro
import matplotlib.pyplot as plt
import copy
from . import utils
from .lsf import GaussianLsf, GaussHermiteLsf
import traceback
import dill as pickle
try:
    import __builtin__ as builtins # Python 2
except ImportError:
    import builtins # Python 3
    
# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

cspeed = 2.99792458e5  # speed of light in km/s

# LSF class dictionary
lsfclass = {'gaussian': GaussianLsf, 'gauss-hermite': GaussHermiteLsf}

# Get print function to be used locally, allows for easy logging
print = utils.getprintfunc() 


def continuum(spec,norder=6,perclevel=90.0,binsize=0.1,interp=True):
    """
    Measure the continuum of a spectrum.

    Parameters
    ----------
    spec : Spec1D object
           A spectrum object.  This at least needs
                to have a FLUX and WAVE attribute.
    norder : float, optional
            Polynomial order to use for the continuum fitting.
            The default is 6.
    perclevel : float, optional
            Percent level to use for the continuum value
            in large bins.  Default is 90.
    binsize : float, optional
            Fraction of the wavelength range (scaled to -1 to +1) to bin.
            Default is 0.1.
    interp : bool, optional
            Use interpolation of the binned values instead of a polynomial
            fit.  Default is True.

    Returns
    -------
    cont : numpy array
            The continuum array, in the same shape as the input flux.

    Examples
    --------
    .. code-block:: python

         cont = continuum(spec)

    """

    wave = spec.wave.copy().reshape(spec.npix,spec.norder)   # make 2D
    flux = spec.flux.copy().reshape(spec.npix,spec.norder)   # make 2D
    mask = spec.mask.copy().reshape(spec.npix,spec.norder)   # make 2D
    cont = flux.copy()*0.0+1
    for o in range(spec.norder):
        w = wave[:,o].copy()
        wr = [np.min(w),np.max(w)]
        x = (w-np.mean(wr))/dln.valrange(wr)*2    # -1 to +1
        y = flux[:,o].copy()
        m = mask[:,o].copy()
        gpix, = np.where((~m) & np.isfinite(x) & np.isfinite(y))
        if len(gpix)==0:
            continue
        # Divide by median
        medy = np.nanmedian(y[gpix])
        if medy <= 0.0:
            medy = 1.0
        y[gpix] /= medy
        # Perform sigma clipping out large positive outliers
        coef = dln.poly_fit(x[gpix],y[gpix],2,robust=True)
        sig = dln.mad(y[gpix]-dln.poly(x[gpix],coef))
        bd1,nbd = dln.where((y[gpix]-dln.poly(x[gpix],coef)) > 5*sig)
        if nbd>0:
            bd = gpix[bd1]
            m[bd] = True
        gdmask = (y>0) & (m==False)        # need positive fluxes and no mask set          
        if np.sum(gdmask)==0:
            continue
        # Bin the data points
        xr = [np.nanmin(x),np.nanmax(x)]
        bins = int(np.ceil((xr[1]-xr[0])/binsize))
        ybin, bin_edges, binnumber = bindata.binned_statistic(x[gdmask],y[gdmask],statistic='percentile',
                                                              percentile=perclevel,bins=bins,range=None)
        xbin = bin_edges[0:-1]+0.5*binsize
        # Interpolate to full grid
        if interp is True:
            fnt = np.isfinite(ybin)
            cont1 = dln.interp(xbin[fnt],ybin[fnt],x,kind='quadratic',extrapolate=True,exporder=1)
        else:
            coef = dln.poly_fit(xbin,ybin,norder)
            cont1 = dln.poly(x,coef)
        cont1 *= medy
        cont[:,o] = cont1
        
    # Flatten to 1D if norder=1
    if spec.norder==1:
        cont = cont.flatten()            
        
    return cont


# Combine multiple spectra
def combine(speclist,wave=None,sum=False):
    """
    This function combines/stacks multiple spectra.

    Parameters
    ----------
    speclist : list of Spec1D objects
         The list of spectra to combine.
    wave : array, optional
        The output wavelength array.  By default, the wavelength array of
        the first spectrum is used.
    sum : bool, optional
        Sum the spectra instead of averaging them.  The default is to average.

    Returns
    -------
    spec : Spec1D object
        The final combined/stacked spectrum.

    Examples
    --------
    .. code-block:: python

        spec = combine([spec1,spec2,spec3])

    """
    
    if type(speclist) is list:
        nspec = len(speclist)
        spec1 = speclist[0]
    else:
        nspec = 1
        spec1 = speclist

    # Only one spectrum input but no wavelength, just return the spectrum
    if (nspec==1) & (wave is None):
        return speclist
    
    # Final wavelength scale, if not input then use wavelengths of first spectrum
    if wave is None:
        wave = spec1.wave

    # How many orders in output wavelength
    if (wave.ndim==1):
        nwpix = len(wave)
        nworder = 1
    else:
        nwpix,nworder = wave.shape
            
    # Loop over the spectra
    flux = np.zeros((nwpix,nworder,nspec),float)
    err = np.zeros((nwpix,nworder,nspec),float)
    mask = np.zeros((nwpix,nworder,nspec),bool)
    for i in range(nspec):
        if (nspec==1):
            spec = speclist
        else:
            spec = speclist[i]
        # Interpolate onto final wavelength scale
        spec2 = spec.interp(wave)
        flux[:,:,i] = spec1.flux
        err[:,:,i] = spec1.err
        mask[:,:,i] = spec1.mask
    # Weighted average
    wt = 1.0 / np.maximum(err,0.0001)**2
    aflux = np.sum(flux*wt,axis=2)/ np.sum(wt,axis=2)
    # uncertainty in weighted mean is 1/sqrt(sum(wt))
    # https://physics.stackexchange.com/questions/15197/how-do-you-find-the-uncertainty-of-a-weighted-average
    aerr = 1.0 / np.sqrt(np.sum(wt,axis=2))
    # mask is set only if all mask values are set
    #  and combine
    amask = np.zeros((nwpix,nworder),bool)
    for i in range(nspec):
        amask = np.logical_and(amask,mask[:,:,i])

    # Sum or average?
    if sum is True:
        aflux *= nspec
        aerr *= nspec

    # Create output spectrum object
    ospec = Spec1D(aflux,err=aerr)
    # cannot handle LSF yet

    return ospec


# Object for representing 1D spectra
class Spec1D(object):
    """
    A class to represent 1D spectra.

    Parameters
    ----------
    flux : array or list
        Array of flux values.  Can 2D if there are multiple orders, e.g. [Npix,Norder].
        If the orders do not have the same size, you can pass a list of arrays.
    err : array, optional
        Array of uncertainties in the flux array.  Same shape as flux.
    wave : array, optional
        Array of wavelengths for the flux array.  Same shape as flux.
    mask : boolean array, optional
        Boolean bad pixel mask array (good=False, bad=True) for flux.  Same shape as flux.
    bitmask : array, optional
        Bit mask giving specific information for each flux pixel.  Same shape as flux.
    lsfpars : array, optional
        The Line Spread Profile (LSF) coefficients as a function of wavelength or pixel
        (that is specified in lsfxtype).
    lsftype : string, optional
       The type of LSF: 'Gaussian' or 'Gauss-Hermite'.
    lsfxtype: string, optional
       If lsfpars is input, then this specifies whether the coeficients depend on wavelength
       ('wave') or pixels ('pixel').  The output LSF parameters (e.g. Gaussian sigma), must
       also use the same units.
    lsfsigma : array, optional
        Array of Gaussian sigma values for each pixel.  Same shape as flux.
    instrument : string, optional
        Name of the instrument on which the spectrum was observed.
    filename : string, optional
        File name of the spectrum.
    wavevac: bool, optional
        Whether the spectrum wavelengths are vacuum wavelengths (wavevac=True) or air wavelengths (wavevac=False).

    """

    
    # Initialize the object
    def __init__(self,flux,err=None,wave=None,mask=None,bitmask=None,head=None,lsfpars=None,
                 lsftype='Gaussian',lsfxtype='Wave',lsfsigma=None,instrument=None,
                 filename=None,wavevac=True):
        """ Initialize Spec1D object."""
        # Add everything to kw dictionary
        kw = {'flux':flux,'err':err,'wave':wave,'mask':mask,'bitmask':bitmask,
              'head':head,'lsfpars':lsfpars,'lsftype':lsftype,'lsfxtype':lsfxtype,
              'lsfsigma':lsfsigma,'instrument':instrument,'filename':filename,
              'wavevac':wavevac}
        # Spec1D object input, overwrite defaults in kw
        if isinstance(flux,Spec1D):
            sp = flux
            kw['flux'] = None
            for k in kw.keys():
                # LSF attributes
                if k.startswith('lsf') and hasattr(sp,'lsf'):
                    klsf = k[3:]
                    if klsf=='sigma': klsf='_sigma'
                    if hasattr(sp.lsf,klsf):
                        kw[k] = copy.deepcopy(getattr(sp.lsf,klsf))
                # Spec1D attributes
                else:
                    if hasattr(sp,k):
                        kw[k] = copy.deepcopy(getattr(sp,k))
            # Get LSF class name
            for k in lsfclass.keys():
                if sp.lsf.__class__ == lsfclass[k]:
                    kw['lsftype'] = k
        # If norder==1, then all arrays will be 1-D.
        # Figure out orders
        numpix,norder,dtype = self._info_from_input(kw['flux'])
        # Check the wavelengths for masked pixels if a 2D array was input
        if kw['wave'] is not None and type(kw['wave']) is np.ndarray:
            numpix = np.zeros(norder,int)
            for i in range(norder):
                if norder>1 or wave.ndim==2:
                    gdpix, = np.where((kw['wave'][:,i]>0) & np.isfinite(kw['wave'][:,i]))
                else:
                    gdpix, = np.where((kw['wave']>0) & np.isfinite(kw['wave']))                    
                if len(gdpix)==0:
                    warnings.warn('All masked pixels in order '+str(i))
                    numpix[i] = 0
                else:
                    numpix[i] = np.max(gdpix)+1
        self.numpix = numpix
        self.flux = self._merge_multiorder_data(kw['flux'],missing_value=0.0)
        if kw['err'] is not None:
            self.err = self._merge_multiorder_data(kw['err'],missing_value=1e30)
            if self.err.shape != self.flux.shape:
                raise ValueError('Error and Flux array sizes do not match')
        else:
            self.err = kw['err']
        if kw['wave'] is not None:
            self.wave = self._merge_multiorder_data(kw['wave'],missing_value=0.0)
            if self.wave.shape != self.flux.shape:
                raise ValueError('Wave and Flux array sizes do not match')            
        else:
            self.wave = kw['wave']
        if kw['mask'] is not None:
            self.mask = self._merge_multiorder_data(kw['mask'],missing_value=True)
            if self.mask.shape != self.flux.shape:
                raise ValueError('Mask and Flux array sizes do not match')            
        if kw['mask'] is None:
            self.mask = np.zeros(kw['flux'].shape,bool)
            if norder==1:
                self.mask = np.squeeze(self.mask)
        if kw['bitmask'] is not None:
            self.bitmask = self._merge_multiorder_data(kw['bitmask'],missing_value=0)
            if self.bitmask.shape != self.flux.shape:
                raise ValueError('Bitmask and Flux array sizes do not match')
        else:
            self.bitmask = kw['bitmask']
        self.head = kw['head']
        if kw['lsftype'].lower() not in lsfclass.keys():
            raise ValueError(kw['lsftype']+' not supported yet')
        self.lsf = lsfclass[kw['lsftype'].lower()](wave=kw['wave'],pars=kw['lsfpars'],
                                                   xtype=kw['lsfxtype'],lsftype=kw['lsftype'],
                                                   sigma=kw['lsfsigma'])
        self.instrument = kw['instrument']
        self.filename = kw['filename']
        self._wavevac = kw['wavevac']
        self.normalized = False
        self.continuum_func = continuum
        self._cont = None
        self._child = False
        self.bc = None
        return

    @property
    def size(self):
        """ Return number of total 'good' pixels (not including buffer pixels)."""
        return np.sum(self.numpix)

    @property
    def ndim(self):
        """ Return the number of dimensions."""
        return self.flux.ndim
    
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
        return self.flux.shape
    
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
    
    def __getitem__(self,index):
        """
        Return a slice or subset of spectral orders

        This returns a new Spec1D with arrays/attributes that are passed by REFERENCE.
        This way you can modify the values in the returned object
        and it will modify the original arrays.

        There are four types of inputs
        1) integer - returns a specific order
        2) 1-D slice (single order) - return that slice of the data
        3) 1-D slice (multiple orders) - return multiple orders using slice
        4) 1-D slice + integer - returns slice of data for order index

        For example,
        sp[0].flux[0] = 1000
        print(sp[0].flux[0:2])
        [1.00000000e+03, 1.57173725e-09]
        print(sp.flux[0:2,0])
        [1.00000000e+03 1.57173725e-09]

        User-added SCALAR attributes will be passed on with this method, but ARRAY/LIST
        attributes will *NOT*.
        """

        # Four types of inputs
        # 1) integer - returns a specific order
        # 2) 1-D slice (single order) - return that slice of the data
        # 3) 1-D slice (multiple orders) - return multiple orders using slice
        # 4) 1-D slice + integer - returns slice of data for order index

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

        # Use a slice because that returns the data by reference
        #   using an index will return the data by value

        if self.norder==1:
            flux = self.flux[slc]
        else:
            flux = self.flux[slc,order]
        if norder==1: flux = flux.flatten()
        # keyword parameters
        kwargs = {}
        for c in ['err','wave','mask','bitmask']:
            if hasattr(self,c) and getattr(self,c) is not None:
                if self.norder==1:
                    kwargs[c] = getattr(self,c)[slc]
                else:
                    kwargs[c] = getattr(self,c)[slc,order]
                if norder==1: kwargs[c] = kwargs[c].flatten()
        # LSF keyword parameters
        lsf = self.lsf[index]
        kwargs['lsfpars'] = lsf.pars
        kwargs['lsftype'] = lsf.lsftype
        kwargs['lsfxtype'] = lsf.xtype
        kwargs['lsfsigma'] = lsf._sigma
        # Other keywords
        for c in ['head','instrument','filename','wavevac']:
            kwargs[c] = getattr(self,c)
            
        # Make the spectrum
        ospec = Spec1D(flux,**kwargs)
        
        # Add extras
        if hasattr(self,'_cont') and getattr(self,'_cont') is not None:
            if self.norder==1:
                ospec._cont = self._cont[slc]
            else:
                ospec._cont = self._cont[slc,order]                
        ospec._child = True
        if self.bc is not None:
            ospec.bc = self.bc
        # Add user-added scalar attributes
        sdict,vdict = self._userattributes
        for n in sdict:
            setattr(ospec,n,sdict[n])
        return ospec

    def __add__(self, value):
        newspec = self.copy()
        if isinstance(value,Spec1D):
            # could interpolate and mask if they don't have the same shape
            if self.shape != value.shape:
                raise ValueError('Shapes do not match')
            newspec.flux += value.flux
            newspec.err = np.sqrt(newspec.err**2 + value.err**2)
            newspec.mask = np.bitwise_or.reduce((newspec.mask,value.mask))
        else:
            newspec.flux += value
        return newspec
        
    def __iadd__(self, value):
        if isinstance(value,Spec1D):
            if self.shape != value.shape:
                raise ValueError('Shapes do not match')            
            self.flux += value.flux
        else:
            self.flux += value
        return self
        
    def __radd__(self, value):
        return self + value
        
    def __sub__(self, value):
        newspec = self.copy()
        if isinstance(value,Spec1D):
            if self.shape != value.shape:
                raise ValueError('Shapes do not match')
            newspec.flux -= value.flux
            newspec.err = np.sqrt(newspec.err**2 + value.err**2)
            newspec.mask = np.bitwise_or.reduce((newspec.mask,value.mask))
        else:
            newspec.flux -= value
        return newspec
              
    def __isub__(self, value):
        if isinstance(value,Spec1D):
            if self.shape != value.shape:
                raise ValueError('Shapes do not match')            
            self.flux -= value.flux
        else:
            self.flux -= value
        return self
         
    def __rsub__(self, value):
        return self - value        
    
    def __mul__(self, value):
        newspec = self.copy()
        if isinstance(value,Spec1D):
            if self.shape != value.shape:
                raise ValueError('Shapes do not match')
            newspec.flux *= value.flux
            newspec.err *= value.flux
            newspec.mask = np.bitwise_or.reduce((newspec.mask,value.mask))
        else:
            newspec.flux *= value
            newspec.err *= value            
        return newspec
               
    def __imul__(self, value):
        if isinstance(value,Spec1D):
            if self.shape != value.shape:
                raise ValueError('Shapes do not match')            
            self.flux *= value.flux
            self.err *= value.flux          
        else:
            self.flux *= value
            self.err *= value            
        return self
    
    def __rmul__(self, value):
        return self * value
               
    def __truediv__(self, value):
        newspec = self.copy()
        if isinstance(value,Spec1D):
            if self.shape != value.shape:
                raise ValueError('Shapes do not match')
            newspec.flux /= value.flux
            newspec.err /= value.flux
            newspec.mask = np.bitwise_or.reduce((newspec.mask,value.mask))
        else:
            newspec.flux /= value
            newspec.err /= value            
        return newspec
      
    def __itruediv__(self, value):
        if isinstance(value,Spec1D):
            if self.shape != value.shape:
                raise ValueError('Shapes do not match')            
            self.flux /= value.flux
            self.err /= value.flux            
        else:
            self.flux /= value
            self.err /= value            
        return self
      
    def __rtruediv__(self, value):
        return self / value

    @property
    def wavevac(self):
        """ Whether wavelengths are in vacuum units."""
        return self._wavevac
    
    @wavevac.setter
    def wavevac(self,wavevac):
        """ Set/change wavelength wavevac value."""
        # Convert wavelength from air->vacuum or vice versa
        if self._wavevac != wavevac:
            for i in range(self.norder):
                npix = self.numpix[i]
                if self.ndim==1:
                    w = self.wave[0:npix]
                else:
                    w = self.wave[0:npix,i]                    
                # Air -> Vacuum
                if wavevac is True:
                    wnew = astro.airtovac(w)
                # Vacuum -> Air
                else:
                    wnew = astro.vactoair(w)
                if self.ndim==1:
                    self.wave[0:npix] = wnew
                else:
                    self.wave[0:npix,i] = wnew
            # change the wavevac value
            self._wavevac = wavevac
                    
    def _info_from_input(self,data):
        """ Figure out the number of orders, number of pixels for each order, and data type for input multi-order data."""
        if type(data) in [list,tuple]:
            if type(data[0]) in (list,tuple,np.ndarray):
                norder = len(data)
                npix = np.zeros(norder,int)                
                for d in data:
                    npix[i] = np.array(d).size
                return npix,norder,np.array(data[0][0]).dtype
            else:
                return [np.array(data).size],1,np.array(data[0]).dtype
        elif type(data) is np.ndarray:
            if data.ndim==1:
                return [data.size],1,data.dtype
            else:
                return data.shape[1]*[data.shape[0]],data.shape[1],data.dtype
        else:
            raise ValueError('Data type '+type(data)+' not supported')
        
    def _merge_multiorder_data(self,data,missing_value=None):
        """ Combine multi-order data into a single array with buffer (if necessary)."""
        numpix,norder,dtype = self._info_from_input(data)
        npix = np.max(numpix)
        if norder==1 or type(data) is np.ndarray:
            out = data
        else:
            # list or tuple
            out = np.zeros((npix,norder),dtype)
            if missing_value is not None:
                out += missing_value   # default value
            for i in range(norder):
                d = np.array(data[i],dtype)
                out[0:len(d),i] = d
        # If norder=1, then return 1-D array
        if norder==1:
            out = np.squeeze(out)
        return out

    def __array__(self):
        return self.flux
    
    def __repr__(self):
        """ Print out the string representation of the Spec1D object."""
        s = repr(self.__class__)+"\n"
        if self.instrument is not None:
            s += self.instrument+" spectrum\n"
        if self.filename is not None:
            s += "File = "+self.filename+"\n"
        if self.snr is not None:
            s += ("S/N = %7.2f" % self.snr)+"\n"
        if self.norder>1:
            s += 'Dimensions: ['+str(self.npix)+','+str(self.norder)+']\n'
        else:
            s += 'Dimensions: ['+str(self.npix)+']\n'                                                       
        s += "Flux = "+str(self.flux)+"\n"
        if self.err is not None:
            s += "Err = "+str(self.err)+"\n"
        if self.wave is not None:
            s += "Wave = "+str(self.wave)
        return s

    @property
    def snr(self):
        """ Return the S/N"""
        if self.flux is not None and self.err is not None:
            if self.mask is not None:
                return np.nanmedian(self.flux[~self.mask]/self.err[~self.mask])
            else:
                return np.nanmedian(self.flux/self.err)                
        else:
            return None
    
    @property
    def cont(self,**kwargs):
        """ Return the continuum."""
        if self._cont is None:
            cont = self.continuum_func(self,**kwargs)
            if self.norder==1:
                cont = np.squeeze(cont)
            self._cont = cont
        return self._cont

    @cont.setter
    def cont(self,inpcont):
        """ Set the continuum."""
        self._cont = inpcont
    
    def wave2pix(self,w,extrapolate=True,order=0):
        """
        Convert wavelength values to pixels using the spectrum dispersion
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

             x = spec.wave2pix(w)

        """        
        if self.wave is None:
            raise Exception("No wavelength information")
        return utils.w2p(self[order].wave,w,assume_sorted=False,extrapolate=extrapolate)        
        
    def pix2wave(self,x,extrapolate=True,order=0):
        """
        Convert pixel values to wavelengths using the spectrum wavelength
        solution.

        Parameters
        ----------
        x : array
          Array of pixel values to convert.
        extrapolate : bool, optional
           Extrapolate beyond the boundaries of the wavelength solution,
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

             w = spec.pix2wave(x)

        """        
        if self.wave is None:
            raise Exception("No wavelength information")
        return utils.p2w(self[order].wave,x,assume_sorted=False,extrapolate=extrapolate)        

    def dispersion(self,x=None,xtype='pixels',extrapolate=True,order=0):
        """
        Wavelength dispersion or step (delta wavelength) at a given pixel
        or wavelength.  This is *always* positive.

        Parameters
        ----------
        x : array, optional
           Array of pixel values at which to return the dispersion.
           Default is to return the values for all pixels.
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

             dw = spec.dispersion(x)

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
        return dw
        
    def normalize(self,**kwargs):
        """
        Normalize the spectrum using a specified continuum function.

        Parameters
        ----------
        **kwargs : arguments that are passed to the continuum function.
              
        Returns
        -------
        The flux and err arrays will normalized (divided) by the continuum
        and the continuum saved in cont.  The normalized property is set to
        True.

        Example
        -------
        .. code-block:: python

             spec.normalize()

        """
        if self.normalized is True:
            return

        # child spectrum
        if self._child:
            warnings.warn('Normalizing a child single-order Spec1D object.  Making copies of flux/err')
            self.flux = self.flux.copy()
            if hasattr(self,'err'):
                self.err = self.err.copy()            
            if self._cont is not None:
                self._cont = self._cont.copy()
            
        self._flux = self.flux.copy()  # Save the original
        
        # Use the continuum_func to get the continuum
        cont = self.cont
        if hasattr(self,'err'):
            self.err /= cont
        self.flux /= cont
        self.cont = cont
        self.normalized = True
        return
        
    def interp(self,x=None,xtype='wave',order=None):
        """ Interpolate onto a new wavelength scale and/or shift by a velocity."""
        # if x is 2D and has multiple dimensions and the spectrum does as well
        # (with the same number of dimensions), and order=None, then it is assumed
        # that the input and output orders are "matched".
        
        # Check input xtype
        if (xtype.lower().find('wave')==-1) & (xtype.lower().find('pix')==-1):
            raise ValueError(xtype+' not supported.  Must be wave or pixel')

        # Convert pixels to wavelength
        if (xtype.lower().find('pix')>-1):
            wave = self.pix2wave(x,order=order)
        else:
            wave = x.copy()
        
        # How many orders in output wavelength
        if (wave.ndim==1):
            nwpix = len(wave)
            nworder = 1
        else:
            nwpix,nworder = wave.shape
        wave = wave.reshape(nwpix,nworder)  # make 2D for order indexing
            
        # Loop over orders in final wavelength
        oflux = np.zeros((nwpix,nworder),float)
        oerr = np.zeros((nwpix,nworder),float)
        omask = np.zeros((nwpix,nworder),bool)
        osigma = np.zeros((nwpix,nworder),float)
        if hasattr(self,'_cont') and self._cont is not None:
            hascont = True
            ocont = np.zeros((nwpix,nworder),float)
        else:
            hascont = False
        for i in range(nworder):
            # Interpolate onto the final wavelength scale
            wave1 = wave[:,i]
            wr1 = dln.minmax(wave1)
            
            # Make spectrum arrays 2D for order indexing, [Npix,Norder]
            swave = self.wave.reshape(self.npix,self.norder)
            sflux = self.flux.reshape(self.npix,self.norder)            
            serr = self.err.reshape(self.npix,self.norder)
            # convert mask to integer 0 or 1
            if hasattr(self,'mask'):
                smask = np.zeros((self.npix,self.norder),int)                
                smask_bool = self.err.reshape(self.npix,self.norder)            
                smask[smask_bool==True] = 1
            else:
                smask = np.zeros((self.npix,self.norder),int)
            if hascont:
                scont = self._cont.reshape(self.npix,self.norder)
                
            # The orders are "matched", one input for one output order
            if (nworder==self.norder) & (order is None):
                swave1 = swave[:,i]                
                sflux1 = sflux[:,i]
                serr1 = serr[:,i]
                ssigma1 = self.lsf.sigma(order=i)
                smask1 = smask[:,i]
                if hascont:
                    scont1 = scont[:,i]
                
                # Some overlap
                if (np.min(swave1)<wr1[1]) & (np.max(swave1)>wr1[0]):
                    # Fix NaN pixels
                    bd,nbd = dln.where(np.isfinite(sflux1)==False) 
                    if nbd>0:
                        sflux1[bd] = 1.0
                        serr1[bd] = 1e30
                        smask1[bd] = 1
                    ind,nind = dln.where( (wave1>np.min(swave1)) & (wave1<np.max(swave1)) )
                    oflux[ind,i] = dln.interp(swave1,sflux1,wave1[ind],extrapolate=False,
                                              assume_sorted=False)
                    oerr[ind,i] = dln.interp(swave1,serr1,wave1[ind],extrapolate=False,
                                             assume_sorted=False,kind='linear')
                    osigma[ind,i] = dln.interp(swave1,ssigma1,wave1[ind],extrapolate=False,
                                               assume_sorted=False)
                    if hascont:
                        ocont[ind,i] = dln.interp(swave1,scont1,wave1[ind],extrapolate=False,
                                                  assume_sorted=False)
                    # Gauss-Hermite, convert to wavelength units
                    if self.lsf.lsftype=='Gauss-Hermite':
                        sdw1 = np.abs(swave1[1:]-swave1[0:-1])
                        sdw1 = np.hstack((sdw1,sdw1[-1])) 
                        dw = dln.interp(swave1,sdw1,wave1[ind],extrapolate=False,assume_sorted=False)
                        osigma[ind,i] *= dw   # in Ang
                    mask_interp = dln.interp(swave1,smask1,wave1[ind],extrapolate=False,assume_sorted=False)                    
                    mask_interp_bool = np.zeros(nind,bool)
                    mask_interp_bool[mask_interp>0.4] = True
                    omask[ind,i] = mask_interp_bool
                    
            # Loop over all spectrum orders
            else:
                # Loop over spectrum orders
                for j in range(self.norder):
                    swave1 = swave[:,j]
                    sflux1 = sflux[:,j]
                    serr1 = serr[:,j]
                    ssigma1 = self.lsf.sigma(order=j)                    
                    smask1 = smask[:,j]
                    if hascont:
                        scont1 = scont[:,j]
                    # Some overlap
                    if (np.min(swave1)<wr1[1]) & (np.max(swave1)>wr1[0]):
                        ind,nind = dln.where( (wave1>np.min(swave1)) & (wave1<np.max(swave1)) )
                        oflux[ind,i] = dln.interp(swave1,sflux1,wave1[ind],extrapolate=False,
                                                  assume_sorted=False)
                        oerr[ind,i] = dln.interp(swave1,serr1,wave1[ind],extrapolate=False,
                                                 assume_sorted=False,kind='linear')
                        osigma[ind,i] = dln.interp(swave1,ssigma1,wave1[ind],extrapolate=False,
                                                   assume_sorted=False)
                        if hascont:
                            ocont[ind,i] = dln.interp(swave1,scont1,wave1[ind],extrapolate=False,
                                                      assume_sorted=False)
                        mask_interp = dln.interp(swave1,smask1,wave1[ind],extrapolate=False,
                                                 assume_sorted=False)                    
                        mask_interp_bool = np.zeros(nind,bool)
                        mask_interp_bool[mask_interp>0.4] = True
                        omask[ind,i] = mask_interp_bool
                    # Currently this does NOT deal with the overlap of multiple orders (e.g. averaging)
        # Flatten if 1D
        if (x.ndim==1):
            wave = wave.flatten()
            oflux = oflux.flatten()
            oerr = oerr.flatten()
            osigma = osigma.flatten()            
            omask = omask.flatten()
            if hascont:
                ocont = ocont.flatten()
        # Create output spectrum object
        if self.lsf.lsftype=='Gauss-Hermite':
            # Can't interpolate Gauss-Hermite LSF yet
            # instead convert to a Gaussian approximation in wavelength units
            #print('Cannot interpolate Gauss-Hermite LSF yet')
            lsfxtype = 'wave'
        else:
            lsfxtype = self.lsf.xtype
        ospec = Spec1D(oflux,wave=wave,err=oerr,mask=omask,lsftype='Gaussian',
                       lsfxtype=lsfxtype,lsfsigma=osigma)
        if hascont:
            ospec._cont = ocont

        return ospec

    def flatten(self):
        """ Flatten the arrays."""
        # This flattens multiple orders and stacks them into one long spectrum
        if self.norder==1:
            return self.copy()
        flux = np.zeros(self.size)
        wave = np.zeros(self.size)
        mask = np.ones(self.size,bool)        
        if self.err is not None:
            err = np.zeros(self.size)
        else:
            err = None
        if self.bitmask is not None:
            bitmask = np.zeros(self.size,self.bitmask.dtype)
        else:
            bitmask = None
        if self.lsf._sigma is not None:
            lsfsigma = np.zeros(self.size)
        else:
            lsfsigma = None
        # Loop over orders
        cnt = 0
        for sp in self:
            slc = slice(cnt,cnt+sp.npix)
            flux[slc] = sp.flux
            if err is not None:
                err[slc] = sp.err
            wave[slc] = sp.wave
            mask[slc] = sp.mask
            if bitmask is not None:
                bitmask[slc] = sp.bitmask
            if lsfsigma is not None:
                lsfsigma[slc] = sp.lsf._sigma
            cnt += sp.npix
        lsfpars = self[0].lsf.pars
        # Instantiate the new spec1d object
        ospec = Spec1D(flux,err=err,mask=mask,wave=wave,bitmask=bitmask,lsfpars=lsfpars,
                       lsftype=self.lsf.lsftype,lsfxtype=self.lsf.xtype,lsfsigma=lsfsigma,head=self.head,
                       instrument=self.instrument,filename=self.filename,wavevac=self.wavevac)
        # If the LSF is in pixel units, then it won't work right.
        ospec._child = True
        return ospec

    def trim(self,w0,w1):
        """
        Trim the spectrum in wavelength.

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

        spec.trim(5500.0,6500.0)

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
        if len(blankorder)>0:
            self.remove_order(blankorder)
        ranges = ranges[goodorder,:]
        numpix = list(ranges[:,1]-ranges[:,0]+1)
        npix = np.max(numpix)    # new npix        
        if len(goodorder)==1:
            ranges = ranges.flatten()
        # Trim the remaining orders
        default = {'flux':0.0,'err':1e30,'wave':0.0,'mask':True,'bitmask':0,'_cont':1.0}
        for c in ['flux','err','wave','mask','bitmask','_cont']:
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
        # Trim the lsf
        self.lsf.trim(w0,w1)

    def rebin(self,nbin,mean=False):
        """
        Bin the data spectrally.
        """
        # Only keep complete bins and remove any excess
        newnpix = self.npix//nbin
        numpix = self.numpix.copy()
        # Bin the data
        default = {'flux':0.0,'err':1e30,'wave':0.0,'mask':True,'bitmask':0,'_cont':1.0}
        for c in ['flux','err','wave','mask','bitmask','_cont']:
            if hasattr(self,c) and getattr(self,c) is not None:
                # Type of binning
                kwargs = {'tot':False,'bitwiseor':False}                
                if c in ['wave','mask']:
                    kwargs['tot'] = False
                elif c=='bitmask':
                    kwargs['bitwiseor'] = True
                elif c=='err':
                    kwargs['tot'] = True
                else:  # flux and _cont
                    kwargs['tot'] = (not mean)
                # Do the rebinning
                arr = getattr(self,c)                
                if self.norder==1:
                    arr = np.atleast_2d(arr).T
                newarr = np.zeros([newnpix,self.norder],arr.dtype)
                newarr[:,:] = default[c]
                for o in range(self.norder):
                    arr1 = arr[:self.numpix[o],o]   # trim to good values
                    if c=='err': arr1 = arr1**2
                    newvals = dln.rebin(arr1,binsize=nbin,**kwargs)
                    if c=='err':   # add err in quadrature
                        newvals = np.sqrt(newvals)
                        if mean: newvals /= nbin                            
                    newarr[:len(newvals),o] = newvals
                    numpix[o] = len(newvals)
                if self.norder==1:
                    newarr = newarr.flatten()
                setattr(self,c,newarr)
        self.numpix = numpix
        # Bin the lsf
        self.lsf.rebin(nbin,mean=mean)
        
    def prepare(self,spec,norm=None,continuum_func=None):
        """
        Return a copy of this (synthetic) spectrum convolved to the input LSF and wavelength arrays.

        Parameters
        ----------
        sp : Spec1D object
           Spec1D object to use to prepare the spectrum.
        norm : boolean, optional
           Normalize the spectrum.  Default is match the normalization state
             of the input spectrum.
        continuum_func : function, optional
           The continuum function to use.

        Returns
        -------
        pspec : Spec1D object
           The prepared spectrum.

        Example
        -------

        pspec = spec.prepare(lsf)

        """
        # Convolve with LSF and do air<->vacuum wavelength conversion
        
        # Check some things
        if self.norder>1:
            raise ValueError('Can only prepare a single order synthetic spectrum')
        if hasattr(spec,'wave')==False:
            raise ValueError('spec object must have wave array')
        if hasattr(spec,'lsf')==False:
            raise ValueError('spec object must have lsf property')
        lsf = spec.lsf
        
        # Temporary working copy of this spectrum
        tempspec = self.copy()

        # Only interpolate errors if they are not all zero
        if self.err is not None and np.sum(self.err) != 0.0:
            doerr = True
        else:
            doerr = False
            
        # Make sure they are using the same type of wavelengths
        # Convert wavelength from air->vacuum or vice versa
        tempspec.wavevac = spec.wavevac   # will automatically convert behind the scenes

        # Initialize the output spectrum
        if lsf.wave.ndim==2:
            npix,norder = lsf.wave.shape
        else:
            npix = len(lsf.wave)
            norder = 1
        pspec = Spec1D(np.zeros((npix,norder),np.float32),err=np.zeros((npix,norder),np.float32),
                       wave=lsf.wave.copy(),lsftype=lsf.lsftype,lsfxtype=lsf.xtype)
        pspec.lsf = lsf.copy()
        hascont = hasattr(tempspec,'_cont') and tempspec._cont is not None
        if hascont:
            pspec._cont = np.zeros((npix,norder),np.float32)
            if norder==1:
                pspec._cont = np.squeeze(pspec._cont)
        if continuum_func is not None:
            pspec.continuum_func = continuum_func

        # Check that the requested wavelengths are covered
        if self.wrange[0]>np.min(lsf.wave) or self.wrange[1]<np.max(lsf.wave):
            raise ValueError('Requested wavelength range not fully covered')
            
        # Loop over orders
        wave = lsf.wave.copy()
        for o in range(norder):
            if norder==1:
                wobs = wave
            else:
                wobs = wave[:,o]
            gdw, = np.where(wobs > 0)
            wobs = wobs[gdw]
            npix1 = len(gdw)
            dw = np.median(dln.slope(wobs))
            wv1,ind1 = dln.closest(tempspec.wave,np.min(wobs)-2*np.abs(dw))
            wv2,ind2 = dln.closest(tempspec.wave,np.max(wobs)+2*np.abs(dw))
            outflux = tempspec.flux[ind1:ind2+1]
            outwave = tempspec.wave[ind1:ind2+1]
            if hascont:
                outcont = tempspec._cont[ind1:ind2+1]
            if doerr:
                outerr = tempspec.err[ind1:ind2+1]            
            
            # Rebin, if necessary
            #  get LSF FWHM (A) for a handful of positions across the spectrum
            xp = np.arange(npix1//20)*20
            fwhm = lsf.fwhm(wobs[xp],xtype='Wave',order=o)
            # FWHM is in units of lsf.xtype, convert to wavelength/angstroms, if necessary
            if lsf.xtype.lower().find('pix')>-1:
                fwhm *= np.abs(dw)
            #  convert FWHM (A) in number of model pixels at those positions
            dwmod = dln.slope(outwave)
            dwmod = np.hstack((dwmod,dwmod[-1]))
            xpmod = dln.interp(outwave,np.arange(len(outwave)),wobs[xp],kind='cubic',
                               assume_sorted=False,extrapolate=True)
            xpmod = np.round(xpmod).astype(int)
            fwhmpix = np.abs(fwhm/dwmod[xpmod])
            # need at least ~4 pixels per LSF FWHM across the spectrum
            #  using 3 affects the final profile shape
            nbin = np.round(np.min(fwhmpix)//4).astype(int)
            
            if np.min(fwhmpix) < 3.7:
                cmt = 'Synthetic spectrum has lower resolution than the desired output spectrum. Only '
                cmt += str(np.min(fwhmpix))+' pixels per resolution element'
                warnings.warn(cmt)
            if np.min(fwhmpix) < 2.0:
                cmt = 'Synthetic spectrum has lower resolution than the desired output spectrum. Only '
                cmt += str(np.min(fwhmpix))+' pixels per resolution element'                
                raise Exception(cmt)
            if nbin>1:
                npix2 = np.round(len(outflux) // nbin).astype(int)
                outflux = dln.rebin(outflux[0:npix2*nbin],npix2)
                outwave = dln.rebin(outwave[0:npix2*nbin],npix2)
                if hascont:
                    outcont = dln.rebin(outcont[0:npix2*nbin],npix2)
                if doerr:
                    outerr = np.sqrt(dln.rebin(outerr[0:npix2*nbin]**2,npix2))
                
            # Convolve
            lsf2d = lsf.anyarray(outwave,xtype='Wave',order=o,original=False)
            cflux = utils.convolve_sparse(outflux,lsf2d)
            if doerr:
                cerr = np.sqrt(utils.convolve_sparse(outerr**2,lsf2d))
            # Interpolate onto final wavelength array
            flux = dln.interp(outwave,cflux,wobs,extrapolate=False,assume_sorted=False)
            if hascont:
                cont = dln.interp(outwave,outcont,wobs,extrapolate=False,assume_sorted=False)
            if doerr:
                err = dln.interp(outwave,cerr,wobs,extrapolate=False,assume_sorted=False)            
            if norder>1:
                pspec.flux[0:len(flux),o] = flux
                if hascont:
                    pspec._cont[0:len(cont),o] = cont
                if doerr:
                    pspec.err[0:len(err),o] = err
            else:
                pspec.flux[0:len(flux)] = flux
                if hascont:
                    pspec._cont[0:len(cont)] = cont
                if doerr:
                    pspec.err[0:len(err)] = err
            if npix1 < npix:
                if norder>1:
                    pspec.mask[len(flux):,o] = True
                else:
                    pspec.mask[len(flux):] = True                
            pspec.normalized = False
            
        # Normalize
        if norm or (norm is None and spec.normalized):
            newcont = pspec.continuum_func(pspec)
            pspec.flux /= newcont
            if hascont:
                pspec._cont *= newcont
            if doerr:
                pspec.err /= newcont
            pspec.normalized = True
            
        return pspec
    
    def plot(self,ax=None,c=None,masked=True):
        """ Plot the spectrum dealing with buffer and masked pixels and keeping color constant for all orders."""
        if ax is None:
            ax = plt
            
        """ Plot the spectrum."""
        for i in range(self.norder):
            npix = self.numpix[i]                
            if self.ndim==1:
                w = self.wave[0:npix]
                f = self.flux[0:npix]
                m = self.mask[0:npix]
            else:
                w = self.wave[0:npix,i]
                f = self.flux[0:npix,i]
                m = self.mask[0:npix,i]
            if masked:
                f = f.copy()
                f[m] = np.nan
            
            if i==0:
                line1, = ax.plot(w,f)
            else:
                line, = ax.plot(w,f,c=line1.get_color())              
    
    def copy(self):
        """ Create a new copy."""
        if self.lsf.pars is not None:
            newlsfpars = self.lsf.pars.copy()
        else:
            newlsfpars = None
        newlsf = self.lsf.copy()
        if self.err is not None:
            newerr = self.err.copy()
        else:
            newerr = None
        if self.wave is not None:
            newwave = self.wave.copy()
        else:
            newwave = None
        if self.mask is not None:
            newmask = self.mask.copy()
        else:
            newmask = None
        if self._cont is not None:
            newcont = self._cont.copy()
            if newcont.ndim==2 and newcont.shape[1]==1:
                newcont = newcont.flatten()
        else:
            newcont = None
        new = Spec1D(self.flux.copy(),err=newerr,wave=newwave,mask=newmask,
                     lsfpars=newlsfpars,lsftype=newlsf.lsftype,lsfxtype=newlsf.xtype,
                     lsfsigma=None,instrument=self.instrument,filename=self.filename)
        new._cont = newcont
        new.lsf = newlsf  # make sure all parts of Lsf are copied over
        for name, value in vars(self).items():
            if name not in ['flux','wave','err','mask','lsf','numpix','_cont','instrument','filename']:
                if value is not None:
                    newvalue = copy.deepcopy(value)
                else:
                    newvalue = None
                setattr(new,name,newvalue)
        return new

    def remove_order(self,order):
        """ Remove orders from spectrum."""
        for c in ['flux','err','wave','mask','bitmask','_cont']:
            if hasattr(self,c) and getattr(self,c) is not None:
                setattr(self,c,np.delete(getattr(self,c),order,axis=1))
        self.lsf.remove_order(order)
        self.numpix = np.delete(self.numpix,order)
        # Single order, use 1-D arrays
        if self.norder==1:
            for c in ['flux','err','wave','mask','_cont']:
                if hasattr(self,c) and getattr(self,c) is not None:
                    setattr(self,c,getattr(self,c).flatten())
            
    def add_order(self,norder=1):
        """ Add blank orders to the spectrum."""
        props = ['flux','err','wave','mask','_cont']
        oldnorder = self.norder
        for p in props:
            if hasattr(self,p) and getattr(self,p) is not None:
                old = getattr(self,p)
                oldsh = list(old.shape)
                newsh = oldsh
                newsh[1] += norder
                new = np.zeros(newsh,old.dtype)
                new[:,0:oldnorder] = old  # transfer data
                setattr(self,p,new)
        self.numpix += norder*[0]

    def remove_blank_orders(self,verbose=False):
        """ Remove any blank orders """
        blank = (self.numpix==0)
        if np.sum(~blank)==0:
            print('All orders are blank')
            self.flux = np.array([],float)
            if self.err is not None: self.err = np.array([],float)
            self.wave = np.array([],float)
            if self.mask is not None: self.mask = np.array([],bool)
            if self.bitmask is not None: self.bitmask = np.array([],int)
            if self._cont is not None: self._cont = np.array([],float)
            if self.lsf.wave is not None: self.lsf.wave = np.array([],float)
            if self.lsf.pars is not None: self.lsf.pars = np.array([],float)            
            if self.lsf._sigma is not None: self.lsf._sigma = np.array([],float)
            self.numpix = []
            self.norder = 0
        if np.sum(blank)>0:
            for name in ['flux','err','wave','mask','bitmask','_cont']:
                if hasattr(self,name) and getattr(self,name) is not None:
                    val = getattr(self,name)
                    val = val[:,~blank]
                    if val.ndim==2 and val.shape[1]==1: val=val.flatten()
                    setattr(self,name,val)
            self.numpix = self.numpix[~blank]            
            if self.lsf.wave is not None:
                self.lsf.wave = self.lsf.wave[:,~blank]
            if self.lsf.pars is not None:
                self.lsf.pars = self.lsf.pars[:,~blank]                
            if self.lsf._sigma is not None:
                self.lsf._sigma = self.lsf._sigma[:,~blank]
            badorders, = np.where(blank)
            if verbose:
                print('Removing blank order(s): '+','.join(np.char.array(badorders).astype(str)))
        
    def append(self,newspec):
        """ Append a new spectrum to this one. Basically combining orders."""
        # Make sure the LSFs are compatible same lsftype and xtype
        if hasattr(self,'lsf') != hasattr(newspec,'lsf'):
            raise ValueError('Spectrum and input spectrum both need to have an LSF or both not have it.')
        if hasattr(self,'lsf'):
            if type(self.lsf) is not type(newspec.lsf):
                raise ValueError('Input LSF ('+str(type(newspec.lsf))+') is not compatible with spectrum LSF ('+str(type(self.lsf))+')')
            if self.lsf.xtype != newspec.lsf.xtype:
                raise ValueError('Input LSF xtype ('+self.lsf.xtype+') is not compatible with spectrum LSF xtype ('+self.lsf.xtype+')')
            if hasattr(self.lsf,'pars') != hasattr(newspec.lsf,'pars'):
                raise ValueError('Input LSF and spectrum LSF both need to have a pars array or both not have it.')
        if self.normalized != newspec.normalized:
            raise ValueError('Spectrum and input spectrum do NOT have the same normalization type')
        # Get new maximum npix
        npix = np.max([self.npix,newspec.npix])
        # Get number of orders
        norder = self.norder+newspec.norder
        # Retain a copy of the old spectrum
        origsp = self.copy()
        # Change the arrays
        props = ['flux','err','wave','mask','_cont']
        newshape = [npix,norder]
        for p in props:
            if p != 'mask':
                arr = np.zeros(newshape,float)
            else:
                arr = np.zeros(newshape,bool)
            # Transfer the original data    
            if hasattr(self,p) and getattr(self,p) is not None:
                orig = getattr(origsp,p)
                arr[0:origsp.npix,0:origsp.norder] = orig
            else:
                orig = None
            # Add the new spectral information
            if hasattr(newspec,p) and getattr(newspec,p) is not None:
                new = getattr(newspec,p)
                arr[0:newspec.npix,origsp.norder:] = new.copy()
            else:
                new = None
            # Set the new property
            if orig is not None and new is not None:
                setattr(self,p,arr)
        # Deal with LSF stuff
        if hasattr(self,'lsf'):
            origlsf = self.lsf.copy()
            self.lsf.wave = self.wave
            # Combine pars
            if hasattr(self.lsf,'pars') and hasattr(newspec.lsf,'pars'):
                origpars = origlsf.pars
                newlsfshape = [np.max([origpars.shape[0],newspec.lsf.pars.shape[0]]),norder]
                newpars = np.zeros(newlsfshape,float)
                # lowest power polycoefficients first, any extra ones at the end
                newpars[0:origpars.shape[0],0:origsp.norder] = origpars
                newpars[0:newspec.lsf.pars.shape[0]:,origsp.norder:] = newspec.lsf.pars.copy()
                self.lsf.pars = newpars
            # Combine _sigma
            if (hasattr(self.lsf,'_sigma') and getattr(self.lsf,'_sigma') is not None and
                hasattr(newspec.lsf,'_sigma') and getattr(newspec.lsf,'_sigma') is not None):
                newsigma = np.zeros(newshape,float)
                newsigma[0:origsp.npix,0:origsp.norder] = origsp.lsf._sigma
                newsigma[0:newspec.npix,origsp.norder:] = newspec.lsf._sigma.copy()
                
        self.numpix = np.concatenate((self.numpix,newspec.numpix))
        del origsp
        
    def barycorr(self):
        """ calculate the barycentric correction."""
        # keck = EarthLocation.of_site('Keck')  # the easiest way... but requires internet
        #keck = EarthLocation.from_geodetic(lat=19.8283*u.deg, lon=-155.4783*u.deg, height=4160*u.m)
        # Calculate the barycentric correction
        if hasattr(self,'bc') is False or self.bc is None:
            if hasattr(self,'observatory') is False:
                print('No observatory information.  Cannot calculate barycentric correction.')
                return 0.0
            obs = EarthLocation.of_site(self.observatory)
            ra = self.head.get('ra')
            if type(ra) is str:
                ra = dln.sexig2ten(ra)*15.0  # convert to degrees
            dec = self.head.get('dec')
            if type(dec) is str:
                dec = dln.sexig2ten(dec)
            if (ra is None) | (dec is None):
                print('No RA/DEC information in header.  Cannot calculate barycentric correction.')
                return 0.0                
            sc = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
            dateobs = self.head.get('date-obs')
            if (dateobs is None):
                print('No DATE information in header.  Cannot calculate barycentric correction.')
                return 0.0
            t = Time(dateobs, format='isot', scale='utc')
            exptime = self.head.get('exptime')
            # Take mid-point of exposure
            if exptime is not None:
                t += 0.5*exptime/3600.0/24.0   # add half in days
            barycorr = sc.radial_velocity_correction(kind='barycentric',obstime=t, location=obs)  
            bc = barycorr.to(u.km/u.s)  
            self.bc = bc.value
        return self.bc

    @property
    def _userattributes(self):
        """ Return a dictionary of user-added attributes.  Scalars and vectors separately"""
        attributes = dir(self)
        sdict = {}   # scalar dictionary
        vdict = {}   # vector dictionary
        # Get all of the "built-in" attributes of the Spec1D object
        #   make "blank" spectrum and check all of the properties
        blank = Spec1D(np.arange(10),wave=np.arange(10))
        ckeys = dir(blank)
        ckeys = [c for c in ckeys if c[0]!='_']
        for a in attributes:
            if a.lower() not in ckeys and a[0]!='_':
                val = getattr(self,a)
                # Scalar attributes
                if dln.size(val)<=1:
                    if val is None:
                        sdict[a] = ['None']
                    else:
                        sdict[a] = [val]   # needs to be a list to construct the table later
                # Arrays
                else:
                    vdict[a] = val
        return sdict,vdict

    def write(self,outfile,overwrite=True):
        """ Write the spectrum to a FITS file."""

        hdu = fits.HDUList()
        # Header
        hdu.append(fits.PrimaryHDU(header=self.head))
        hdu[0].header['COMMENT'] = 'HDU0: header'
        hdu[0].header['COMMENT'] = 'HDU1: flux'
        hdu[0].header['COMMENT'] = 'HDU2: flux error'
        hdu[0].header['COMMENT'] = 'HDU3: wavelength'
        hdu[0].header['COMMENT'] = 'HDU4: mask'
        hdu[0].header['COMMENT'] = 'HDU5: LSF'
        hdu[0].header['COMMENT'] = 'HDU6: continuum'
        hdu[0].header['COMMENT'] = 'HDU7: continuum function'        
        hdu[0].header['SPECTYPE'] = 'SPEC1D'
        hdu[0].header['INSTRMNT'] = self.instrument
        hdu[0].header['WAVEVAC'] = self.wavevac
        hdu[0].header['NORMLIZD'] = self.normalized
        hdu[0].header['NDIM'] = self.ndim
        hdu[0].header['NPIX'] = self.npix
        hdu[0].header['NORDER'] = self.norder
        for i in range(self.norder):
            hdu[0].header['NUMPIX'+str(i+1)] = self.numpix[i]
        if self.bc is not None:
            hdu[0].header['BC'] = (self.bc,'barycentric correction')
        if np.isfinite(self.snr):
            hdu[0].header['SNR'] = (self.snr,'signal-to-noise')
        else:
            hdu[0].header['SNR'] = (str(self.snr),'signal-to-noise')
        # Flux
        hdu.append(fits.ImageHDU(self.flux))
        hdu[1].header['BUNIT'] = 'Flux'
        hdu[1].header['EXTNAME'] = 'FLUX'
        # Error
        hdu.append(fits.ImageHDU(self.err))
        hdu[2].header['BUNIT'] = 'Flux Error'
        hdu[2].header['EXTNAME'] = 'FLUX_ERROR' 
        # Wavelength
        hdu.append(fits.ImageHDU(self.wave))
        hdu[3].header['BUNIT'] = 'Wavelength (Ang)'
        hdu[3].header['EXTNAME'] = 'WAVELENGTH'  
        # Mask
        hdu.append(fits.ImageHDU(self.mask.astype(int)))
        hdu[4].header['BUNIT'] = 'Mask'
        hdu[4].header['EXTNAME'] = 'MASK'
        # LSF
        #  ADD A WRITE() METHOD TO LSF class
        #  can write to FITS or hdu if hdu=True is set
        hdu.append(fits.ImageHDU(self.lsf._sigma))
        hdu[5].header['BUNIT'] = 'LSF'
        hdu[5].header['EXTNAME'] = 'LSF'        
        hdu[5].header['XTYPE'] = self.lsf.xtype
        hdu[5].header['LSFTYPE'] = self.lsf.lsftype
        hdu[5].header['NDIM'] = self.lsf.ndim
        hdu[5].header['NPIX'] = self.lsf.npix
        hdu[5].header['NORDER'] = self.lsf.norder
        # should we put PARS in the header, or should it be a separate extension??
        if self.lsf.pars is not None:
            hdu[5].header['NPARS'] = np.array(self.lsf.pars).size
            lparshape = np.array(self.lsf.pars).shape
            if len(lparshape)>=1:
                hdu[5].header['NPARS1'] = lparshape[0]
            if len(lparshape)==2:
                hdu[5].header['NPARS2'] = lparshape[1]
            for i,p in enumerate(np.array(self.lsf.pars).flatten()):
                hdu[5].header['PAR'+str(i)] = p
        else:
            hdu[5].header['NPARS'] = 0
            hdu[5].header['NPARS1'] = 0
            hdu[5].header['NPARS2'] = 0            
        # Continuum
        hdu.append(fits.ImageHDU(self._cont))
        hdu[6].header['BUNIT'] = 'Continuum'
        hdu[6].header['EXTNAME'] = 'CONTINUUM'
        # Continuum function
        cont_func_ser = pickle.dumps(self.continuum_func)    # serialise the function
        tab = Table(np.atleast_1d(np.array(cont_func_ser)),names=['func'])  # save as FITS binary table
        hdu.append(fits.table_to_hdu(tab))
        hdu[7].header['BUNIT'] = 'Continuum function'
        # Add other attributes to the primary header
        #   these were added by the user
        sdict,vdict = self._userattributes
        nexten = 8
        # Put scalars in a separate table
        if len(sdict)>0:
            tab = Table(sdict)
            hdu.append(fits.table_to_hdu(tab))
            hdu[nexten].header['COMMENT'] = 'Extra scalar attributes'
            hdu[nexten].header['ATTRIBUT'] = True
            hdu[nexten].header['VECTOR'] = False
            hdu[nexten].header['NAME'] = 'scalars'
            nexten += 1
        # Put arrays in separate extensions
        if len(vdict)>0:
            for k in vdict.keys():
                val = vdict[k]
                hdu.append(fits.ImageHDU(val))
                hdu[nexten].header['ATTRIBUT'] = True
                hdu[nexten].header['VECTOR'] = True
                hdu[nexten].header['NAME'] = k           
                nexten += 1
        hdu.verify('fix+warn')
        hdu.writeto(outfile,overwrite=overwrite)
