#!/usr/bin/env python

"""SPEC1D.PY - Create spectrum 1D class and methods

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@noao.edu>'
__version__ = '20180922'  # yyyymmdd                                                                                                                           

import numpy as np
import warnings
from scipy.interpolate import interp1d
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import thecannon as tc
from dlnpyutils import utils as dln, bindata
import copy
from . import utils

# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

cspeed = 2.99792458e5  # speed of light in km/s

# astropy.modeling can handle errors and constraints


# Object for representing LSF (line spread function)
class Lsf:
    # Initalize the object
    def __init__(self,wave=None,pars=None,xtype='wave',lsftype='Gaussian',sigma=None,silent=False):
        # xtype is wave or pixels.  designates what units to use BOTH for the input
        #   arrays to use with PARS and the output units
        if wave is None and xtype=='Wave':
            raise Exception('Need wavelength information if xtype=Wave')
        self.wave = wave
        self.pars = pars
        self.type = lsftype
        self.xtype = xtype
        if wave.ndim==1:
            npix = len(wave)
            norder = 1
        else:
            npix,norder = wave.shape
        self.ndim = wave.ndim
        self.npix = npix
        self.norder = norder
        self._sigma = sigma
        self._array = None
        if (pars is None) & (sigma is None):
            if silent is False: print('No LSF information input.  Assuming Nyquist sampling.')
            # constant FWHM=2.5, sigma=2.5/2.35
            self.pars = np.array([2.5 / 2.35])
            self.xtype = 'Pixels'

    def wave2pix(self,w,extrapolate=True,order=0):
        if self.wave is None:
            raise Exception("No wavelength information")
        if self.ndim==2:
            # Order is always the second dimension
            return utils.w2p(self.wave[:,order],w,extrapolate=extrapolate)            
        else:
            return utils.w2p(self.wave,w,extrapolate=extrapolate)
        
    def pix2wave(self,x,extrapolate=True,order=0):
        if self.wave is None:
            raise Exception("No wavelength information")
        if self.ndim==2:
             # Order is always the second dimension
            return utils.p2w(self.wave[:,order],x,extrapolate=extrapolate)
        else:
            return utils.p2w(self.wave,x,extrapolate=extrapolate)        
        
    # Return FWHM at some positions
    def fwhm(self,x=None,xtype='pixels',order=0):
        #return np.polyval(pars[::-1],x)*2.35
        return self.sigma(x,xtype=xtype,order=order)*2.35
        
    # Return Gaussian sigma
    def sigma(self,x=None,xtype='pixels',order=0,extrapolate=True):
        # The sigma will be returned in units given in lsf.xtype
        if self._sigma is not None:
            _sigma = self._sigma
            if self.ndim==2: _sigma = self._sigma[:,order]
            if x is None:
                return _sigma
            else:
                # Wavelength input
                if xtype.lower().find('wave') > -1:
                    x0 = np.array(x).copy()            # backup
                    x = self.wave2pix(x0,order=order)  # convert to pixels
                # Integer, just return the values
                if( type(x)==int) | (np.array(x).dtype.kind=='i'):
                    return _sigma[x]
                # Floats, interpolate
                else:
                    sig = interp1d(np.arange(len(_sigma)),_sigma,kind='cubic',bounds_error=False,
                                   fill_value=(np.nan,np.nan),assume_sorted=True)(x)
                    # Extrapolate
                    npix = self.npix
                    if ((np.min(x)<0) | (np.max(x)>(npix-1))) & (extrapolate is True):
                        xin = np.arange(npix)
                        # At the beginning
                        if (np.min(x)<0):
                            coef1 = dln.poly_fit(xin[0:10], _sigma[0:10], 2)
                            bd1, nbd1 = dln.where(x <0)
                            sig[bd1] = dln.poly(x[bd1],coef1)
                        # At the end
                        if (np.max(x)>(npix-1)):
                            coef2 = dln.poly_fit(xin[npix-10:], _sigma[npix-10:], 2)
                            bd2, nbd2 = dln.where(x > (npix-1))
                            sig[bd2] = dln.poly(x[bd2],coef2)
                    return sig
                        
        # Need to calculate
        else:
            if x is None:
                x = np.arange(self.npix)
            if self.pars is None:
                   raise Exception("No LSF parameters")
            # Get parameters
            pars = self.pars
            if self.ndim==2: pars=self.pars[:,order]
            # Pixels input
            if xtype.lower().find('pix') > -1:
                # Pixel LSF parameters
                if self.xtype.lower().find('pix') > -1:
                    return np.polyval(pars[::-1],x)
                # Wave LSF parameters
                else:
                    w = self.pix2wave(x,order=order)
                    return np.polyval(pars[::-1],w)                    
            # Wavelengths input
            else:
                # Wavelength LSF parameters
                if self.xtype.lower().find('wave') > -1:
                    return np.polyval(pars[::-1],x)
                # Pixel LSF parameters
                else:
                    x0 = np.array(x).copy()
                    x = self.wave2pix(x0,order=order)
                    return np.polyval(pars[::-1],x)  

    # Clean up bad LSF values
    def clean(self):
        if self._sigma is not None:
            smlen = np.round(self.npix // 50).astype(int)
            if smlen==0: smlen=3
            _sigma = self._sigma.reshape(self.npix,self.norder)   # make 2D
            for o in range(self.norder):
                sig = _sigma[:,o]
                smsig = dln.gsmooth(sig,smlen)
                bd,nbd = dln.where(sig <= 0)
                if nbd>0:
                    sig[bd] = smsig[bd]
                if self.ndim==2:
                    self._sigma[:,o] = sig
                else:
                    self._sigma = sig
                
    # Return actual LSF values
    def vals(self,x):
        # x must be 2D to give x/wavelength CENTERS and the grid on
        #  which to put them
        # or we could have two inputs, xcenter and xgrid
        pass

        # create the LSF array using the input wavelength array input


    # Return full LSF values for the spectrum
    def array(self,order=None):
        # Return what we already have
        if self._array is not None:
            if (self.ndim==2) & (order is not None):
                # [Npix,Nlsf,Norder]
                return self._array[:,:,order]
            else:
                return self._array
        
        ## currently this assumes the LSF parameters use type='Wave'
        #if self.xtype!='Wave' or self.lsftype!='Gaussian':
        #    print('Currently only implemented for xtype=Wave and lsftype=Gaussian')
        #    return

        # Loop over orders and figure out how many Nlsf pixels we need
        #  must be same across all orders
        wave = self.wave.reshape(self.npix,self.norder)
        nlsfarr = np.zeros(norder,dtype=int)
        xsigma = np.zeros((self.npix,norder),dtype=np.float64)
        for o in range(self.norder):
            x = np.arange(self.npix)
            xsig = self.sigma(order=o)
            w = wave[:,o]

            # Convert sigma from wavelength to pixels, if necessary
            if self.xtype.lower().find('wave') > -1:
                wsigma = xsigma.copy()
                dw = dln.slope(w)
                dw = np.hstack((dw,dw[-1]))            
                xsig = wsigma / dw
            xsigma[:,o] = xsig
                
            # Figure out nLSF pixels needed, +/-3 sigma
            nlsf = np.int(np.round(np.max(xsigma)*6))
            if nlsf % 2 == 0: nlsf+=1                   # must be odd
            nlsfarr.append(nlsf)
        nlsf = np.max(np.array(nsflarr))
            
        # Make LSF array
        lsf = np.zeros((self.npix,nlsf,self.norder))
        for o in range(self.norder):
            xsig = xsigma[:,o]
            xlsf = np.arange(nlsf)-nlsf//2
            xlsf2 = np.repeat(xlsf,self.npix).reshape((nlsf,self.npix)).T
            xsigma2 = np.repeat(xsig,nlsf).reshape((self.npix,nlsf))
            lsf[:,:,o] = np.exp(-0.5*xlsf2**2 / xsigma2**2) / (np.sqrt(2*np.pi)*xsigma2)
            # should I use gaussbin????

        # if only one order then reshape
        if self.ndim==1: lsf=lsf.reshape(self.npix,nlsf)
            
        self._array = lsf   # save for next time

        return lsf

    
    # Return full LSF values using contiguous input array
    def anyarray(self,x,xtype='pixels',order=0):

        ## currently this assumes the LSF parameters use type='Wave'
        #if self.xtype!='Wave' or self.lsftype!='Gaussian':
        #    print('Currently only implemented for xtype=Wave and lsftype=Gaussian')
        #    return

        nx = len(x)
        xsigma = self.sigma(x,xtype=xtype,order=order)

        # Get wavelength and pixel arrays
        if xtype.lower().find('pix') > -1:
            w = self.pix2wave(x,order=order)
        else:
            w = x
            
        # Convert sigma from wavelength to pixels, if necessary
        if self.xtype.lower().find('wave') > -1:
            wsigma = xsigma.copy()
            dw = dln.slope(w)
            dw = np.hstack((dw,dw[-1]))            
            xsigma = wsigma / dw

        # Figure out nLSF pixels needed, +/-3 sigma
        nlsf = np.int(np.round(np.max(xsigma)*6))
        if nlsf % 2 == 0: nlsf+=1                   # must be odd
        
        # Make LSF array
        lsf = np.zeros((nx,nlsf))
        xlsf = np.arange(nlsf)-nlsf//2
        xlsf2 = np.repeat(xlsf,nx).reshape((nlsf,nx)).T
        xsigma2 = np.repeat(xsigma,nlsf).reshape((nx,nlsf))
        lsf = np.exp(-0.5*xlsf2**2 / xsigma2**2) / (np.sqrt(2*np.pi)*xsigma2)
        # should I use gaussbin????
        return lsf


    def copy(self):
        """ Create a new copy."""
        return copy.deepcopy(self)

        
    
# Object for representing 1D spectra
class Spec1D:
    # Initialize the object
    def __init__(self,flux,err=None,wave=None,mask=None,lsfpars=None,lsftype='Gaussian',
                 lsfxtype='Wave',lsfsigma=None,instrument=None,filename=None):
        self.flux = flux
        self.err = err
        self.wave = wave
        self.mask = mask
        self.lsf = Lsf(wave=wave,pars=lsfpars,xtype=lsfxtype,lsftype=lsftype,sigma=lsfsigma)
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
        w = self.wave.copy()
        x = (w-np.median(w))/(np.max(w*0.5)-np.min(w*0.5))  # -1 to +1
        y = self.flux.copy()
        gdmask = (y>0)        # need positive fluxes
        ytemp = y.copy()
        # Bin the data points
        xr = [np.nanmin(x),np.nanmax(x)]
        bins = np.ceil((xr[1]-xr[0])/binsize)+1
        ybin, bin_edges, binnumber = bindata.binned_statistic(x,ytemp,statistic='percentile',
                                                              percentile=perclevel,bins=bins,range=None)
        xbin = bin_edges[0:-1]+0.5*binsize
        # Interpolate to full grid
        cont = dln.interp(xbin,ybin,x,extrapolate=True)

        self.flux /= cont
        self.err /= cont
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
