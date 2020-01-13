#!/usr/bin/env python

"""SPEC1D.PY - Create spectrum 1D class and methods

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@noao.edu>'
__version__ = '20180922'  # yyyymmdd                                                                                                                           

import numpy as np
import math
import warnings
from scipy.interpolate import interp1d
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import thecannon as tc
from dlnpyutils import utils as dln, bindata
import copy
from . import utils
from .lsf import GaussianLsf, GaussHermiteLsf

# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

cspeed = 2.99792458e5  # speed of light in km/s

# astropy.modeling can handle errors and constraints

# LSF class dictionary
lsfclass = {'gaussian': GaussianLsf, 'gauss-hermite': GaussHermiteLsf}

# Combine multiple spectra
def combine(speclist,wave=None,sum=False):
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
    flux = np.empty((nwpix,nworder,nspec),float)
    err = np.empty((nwpix,nworder,nspec),float)
    mask = np.empty((nwpix,nworder,nspec),bool)
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
    amask = np.empty((nwpix,nworder),bool)
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
class Spec1D:
    # Initialize the object
    def __init__(self,flux,err=None,wave=None,mask=None,bitmask=None,lsfpars=None,lsftype='Gaussian',
                 lsfxtype='Wave',lsfsigma=None,instrument=None,filename=None):
        self.flux = flux
        self.err = err
        self.wave = wave
        self.mask = mask
        self.bitmask = bitmask
        if lsftype.lower() not in lsfclass.keys():
            raise ValueError(lsftype+' not supported yet')
        self.lsf = lsfclass[lsftype.lower()](wave=wave,pars=lsfpars,xtype=lsfxtype,lsftype=lsftype,sigma=lsfsigma)
        #self.lsf = Lsf(wave=wave,pars=lsfpars,xtype=lsfxtype,lsftype=lsftype,sigma=lsfsigma)
        self.instrument = instrument
        self.filename = filename
        self.snr = None
        if self.err is not None:
            self.snr = np.nanmedian(flux)/np.nanmedian(err)
        self.normalized = False
        if flux.ndim==1:
            npix = len(flux)
            norder = 1
        else:
            npix,norder = flux.shape
        self.ndim = flux.ndim
        self.npix = npix
        self.norder = norder
        return

    def __repr__(self):
        s = repr(self.__class__)+"\n"
        if self.instrument is not None:
            s += self.instrument+" spectrum\n"
        if self.filename is not None:
            s += "File = "+self.filename+"\n"
        if self.snr is not None:
            s += ("S/N = %7.2f" % self.snr)+"\n"
        s += "Flux = "+str(self.flux)+"\n"
        if self.err is not None:
            s += "Err = "+str(self.err)+"\n"
        if self.wave is not None:
            s += "Wave = "+str(self.wave)
        return s

    def wave2pix(self,w,extrapolate=True,order=0):
        if self.wave is None:
            raise Exception("No wavelength information")
        if self.wave.ndim==2:
            # Order is always the second dimension
            return utils.w2p(self.wave[:,order],w,extrapolate=extrapolate)            
        else:
            return utils.w2p(self.wave,w,extrapolate=extrapolate)
        
    def pix2wave(self,x,extrapolate=True,order=0):
        if self.wave is None:
            raise Exception("No wavelength information")
        if self.wave.ndim==2:
             # Order is always the second dimension
            return utils.p2w(self.wave[:,order],x,extrapolate=extrapolate)
        else:
            return utils.p2w(self.wave,x,extrapolate=extrapolate)            

    def normalize(self,ncorder=6,perclevel=0.95):
        self._flux = self.flux  # Save the original
        #nspec, cont, masked = normspec(self,ncorder=ncorder,perclevel=perclevel)

        binsize = 0.05
        perclevel = 90.0
        wave = self.wave.copy().reshape(self.npix,self.norder)   # make 2D
        flux = self.flux.copy().reshape(self.npix,self.norder)   # make 2D
        err = self.err.copy().reshape(self.npix,self.norder)   # make 2D
        cont = err.copy()*0.0
        for o in range(self.norder):
            w = wave[:,o]
            x = (w-np.median(w))/(np.max(w*0.5)-np.min(w*0.5))  # -1 to +1
            y = flux[:,o]
            gdmask = (y>0)        # need positive fluxes
            # Bin the data points
            xr = [np.nanmin(x),np.nanmax(x)]
            bins = np.ceil((xr[1]-xr[0])/binsize)+1
            ybin, bin_edges, binnumber = bindata.binned_statistic(x,y,statistic='percentile',
                                                                  percentile=perclevel,bins=bins,range=None)
            xbin = bin_edges[0:-1]+0.5*binsize
            # Interpolate to full grid
            cont1 = dln.interp(xbin,ybin,x,extrapolate=True)

            flux[:,o] /= cont1
            err[:,o] /= cont1
            cont[:,o] = cont1

        # Flatten to 1D if norder=1
        if self.norder==1:
            flux = flux.flatten()
            err = err.flatten()
            cont = cont.flatten()            

        # Stuff back in
        self.flux = flux
        self.err = err
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
        oflux = np.empty((nwpix,nworder),float)
        oerr = np.empty((nwpix,nworder),float)
        omask = np.empty((nwpix,nworder),bool)
        osigma = np.empty((nwpix,nworder),float)        
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

            # The orders are "matched", one input for one output order
            if (nworder==self.norder) & (order is None):
                swave1 = swave[:,i]                
                sflux1 = sflux[:,i]
                serr1 = serr[:,i]
                ssigma1 = self.lsf.sigma(order=i)                
                smask1 = smask[:,i]
                # Some overlap
                if (np.min(swave1)<wr1[1]) & (np.max(swave1)>wr1[0]):
                    ind,nind = dln.where( (wave1>np.min(swave1)) & (wave1<np.max(swave1)) )
                    oflux[ind,i] = dln.interp(swave1,sflux1,wave1[ind],extrapolate=False,assume_sorted=False)
                    oerr[ind,i] = dln.interp(swave1,serr1,wave1[ind],extrapolate=False,assume_sorted=False)
                    osigma[ind,i] = dln.interp(swave1,ssigma1,wave1[ind],extrapolate=False,assume_sorted=False)
                    mask_interp = dln.interp(swave1,smask1,wave1[ind],extrapolate=False,assume_sorted=False)                    
                    mask_interp_bool = np.empty(nind,bool)
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
                    # Some overlap
                    if (np.min(swave1)<wr1[1]) & (np.max(swave1)>wr1[0]):
                        ind,nind = dln.where( (wave1>np.min(swave1)) & (wave1<np.max(swave1)) )
                        oflux[ind,i] = dln.interp(swave1,sflux1,wave1[ind],extrapolate=False,assume_sorted=False)
                        oerr[ind,i] = dln.interp(swave1,serr1,wave1[ind],extrapolate=False,assume_sorted=False)
                        osigma[ind,i] = dln.interp(swave1,ssigma1,wave1[ind],extrapolate=False,assume_sorted=False)
                        mask_interp = dln.interp(swave1,smask1,wave1[ind],extrapolate=False,assume_sorted=False)                    
                        mask_interp_bool = np.empty(nind,bool)
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
        # Create output spectrum object
        if self.lsf.lsftype=='GaussHermite':
            print('Cannon interpolate Gauss-Hermite LSF yet')
            ospec = Spec1D(oflux,wave=wave,err=oerr,mask=omask)            
        else:
            ospec = Spec1D(oflux,wave=wave,err=oerr,mask=omask,lsftype=self.lsf.lsftype,
                           lsfxtype=self.lsf.xtype,lsfsigma=osigma)

        return ospec
                

    def copy(self):
        """ Create a new copy."""
        new = Spec1D(self.flux,err=self.err,wave=self.wave,mask=self.mask,lsfpars=self.lsf.pars,lsftype=self.lsf.type,
                     lsfxtype=self.lsf.xtype,lsfsigma=self.lsf.sigma,instrument=self.instrument,filename=self.filename)
        new.lsf = copy.deepcopy(self.lsf)  # make sure all parts of Lsf are copied over
        for name, value in vars(self).items():
            if name not in ['flux','wave','err','mask','lsf','instrument','filename']:
                setattr(new,name,copy.deepcopy(value))           

        return new

    def barycorr(self):
        """ calculate the barycentric correction."""
        # keck = EarthLocation.of_site('Keck')  # the easiest way... but requires internet
        #keck = EarthLocation.from_geodetic(lat=19.8283*u.deg, lon=-155.4783*u.deg, height=4160*u.m)
        # Calculate the barycentric correction
        if hasattr(self,'bc') is False:
            if hasattr(self,'observatory') is False:
                print('No observatory information.  Cannot calculate barycentric correction.')
                return None
            obs = EarthLocation.of_site(self.observatory)
            ra = self.head.get('ra')
            dec = self.head.get('dec')  
            if (ra is None) | (dec is None):
                print('No RA/DEC information in header.  Cannot calculate barycentric correction.')
                return None                
            sc = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
            dateobs = self.head.get('date-obs')
            if (dateobs is None):
                print('No DATE information in header.  Cannot calculate barycentric correction.')
                return None
            t = Time(dateobs, format='isot', scale='utc')
            barycorr = sc.radial_velocity_correction(kind='barycentric',obstime=t, location=obs)  
            bc = barycorr.to(u.km/u.s)  
            self.bc = bc
        return self.bc
        
