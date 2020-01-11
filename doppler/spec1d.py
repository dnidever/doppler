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
        
    @staticmethod
    def read(filename=None):
        return rdspec(filename=filename)
    
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

    def rv(self,template):
        """Calculate the RV with respect to a template spectrum"""
        pass
        return
        
    def solve(self):
        """Find the RV and stellar parameters of this spectrum"""
        pass
        return


    def interp(self,wave=None,vel=None):
        """ Interpolate onto a new wavelength scale and/or shift by a velocity."""
        pass

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
        
    
   # maybe add an interp() method to interpolate the
   # spectrum onto a new wavelength scale, outputs a new object

   # a load() or read() method that you can use to read in a spectrum
   # from a file, basically just calls the rdspec() function.
