#!/usr/bin/env python

"""SPEC1D.PY - The spectrum 1D class and methods

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@noao.edu>'
__version__ = '20200112'  # yyyymmdd                                                                                                                           

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
class Spec1D:
    """
    A class to represent 1D spectra.

    Parameters
    ----------
    flux : array
        Array of flux values.  Can 2D if there are multiple orders, e.g. [Npix,Norder].
        The order dimension must always be the last/trailing dimension.
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
    def __init__(self,flux,err=None,wave=None,mask=None,bitmask=None,lsfpars=None,lsftype='Gaussian',
                 lsfxtype='Wave',lsfsigma=None,instrument=None,filename=None,wavevac=True):
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
        self.wavevac = wavevac
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

        Examples
        --------

        x = spec.wave2pix(w)

        """
        
        if self.wave is None:
            raise Exception("No wavelength information")
        if self.wave.ndim==2:
            # Order is always the second dimension
            return utils.w2p(self.wave[:,order],w,extrapolate=extrapolate)            
        else:
            return utils.w2p(self.wave,w,extrapolate=extrapolate)

        
    def pix2wave(self,x,extrapolate=True,order=0):
        """
        Convert pixel values to wavelengths using the spectrum dispersion
        solution.

        Parameters
        ----------
        x : array
          Array of pixel values to convert.
        extrapolate : bool, optional
           Extrapolate beyond the boundaries of the dispersion solution,
           if necessary.  True by default.
        order : int, optional
            The order to use if there are multiple orders.
            The default is 0.
              
        Returns
        -------
        w : array
          The array of converted wavelengths.

        Examples
        --------

        w = spec.pix2wave(x)

        """
        
        if self.wave is None:
            raise Exception("No wavelength information")
        if self.wave.ndim==2:
             # Order is always the second dimension
            return utils.p2w(self.wave[:,order],x,extrapolate=extrapolate)
        else:
            return utils.p2w(self.wave,x,extrapolate=extrapolate)            

        
    def normalize(self,ncorder=6,perclevel=0.95):
        """
        Normalize the spectrum.

        Parameters
        ----------
        ncorder : float, optional
              Polynomial order to use for the continuum fitting.
              The default is 6.
        perclevel : float, optional
              Percent level (1.0 for 100%) to use for the continuum value
              in large bins.  Default is 0.95.
              
        Returns
        -------
        The flux and err arrays will normalized (divided) by the continuum
        and the continuum saved in cont.  The normalized property is set to
        True.

        Examples
        --------

        spec.normalize()

        """

        self._flux = self.flux  # Save the original
        #nspec, cont, masked = normspec(self,ncorder=ncorder,perclevel=perclevel)

        binsize = 0.10
        perclevel = 90.0
        wave = self.wave.copy().reshape(self.npix,self.norder)   # make 2D
        flux = self.flux.copy().reshape(self.npix,self.norder)   # make 2D
        err = self.err.copy().reshape(self.npix,self.norder)     # make 2D
        mask = self.mask.copy().reshape(self.npix,self.norder)   # make 2D
        cont = err.copy()*0.0+1
        for o in range(self.norder):
            w = wave[:,o].copy()
            x = (w-np.median(w))/(np.max(w*0.5)-np.min(w*0.5))  # -1 to +1
            y = flux[:,o].copy()
            m = mask[:,o].copy()
            # Divide by median
            medy = np.nanmedian(y)
            y /= medy
            # Perform sigma clipping out large positive outliers
            coef = dln.poly_fit(x,y,2,robust=True)
            sig = dln.mad(y-dln.poly(x,coef))
            bd,nbd = dln.where((y-dln.poly(x,coef)) > 5*sig)
            if nbd>0: m[bd]=True
            gdmask = (y>0) & (m==False)        # need positive fluxes and no mask set          
            # Bin the data points
            xr = [np.nanmin(x),np.nanmax(x)]
            bins = np.ceil((xr[1]-xr[0])/binsize)+1
            ybin, bin_edges, binnumber = bindata.binned_statistic(x[gdmask],y[gdmask],statistic='percentile',
                                                                  percentile=perclevel,bins=bins,range=None)
            xbin = bin_edges[0:-1]+0.5*binsize
            # Interpolate to full grid
            fnt = np.isfinite(ybin)
            cont1 = dln.interp(xbin[fnt],ybin[fnt],x,extrapolate=True)
            cont1 *= medy
            
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
        oflux = np.zeros((nwpix,nworder),float)
        oerr = np.zeros((nwpix,nworder),float)
        omask = np.zeros((nwpix,nworder),bool)
        osigma = np.zeros((nwpix,nworder),float)        
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
                    # Fix NaN pixels
                    bd,nbd = dln.where(np.isfinite(sflux1)==False) 
                    if nbd>0:
                        sflux1[bd] = 1.0
                        serr1[bd] = 1e30
                        smask1[bd] = 1
                    ind,nind = dln.where( (wave1>np.min(swave1)) & (wave1<np.max(swave1)) )
                    oflux[ind,i] = dln.interp(swave1,sflux1,wave1[ind],extrapolate=False,assume_sorted=False)
                    oerr[ind,i] = dln.interp(swave1,serr1,wave1[ind],extrapolate=False,assume_sorted=False,kind='linear')
                    osigma[ind,i] = dln.interp(swave1,ssigma1,wave1[ind],extrapolate=False,assume_sorted=False)
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
                    # Some overlap
                    if (np.min(swave1)<wr1[1]) & (np.max(swave1)>wr1[0]):
                        ind,nind = dln.where( (wave1>np.min(swave1)) & (wave1<np.max(swave1)) )
                        oflux[ind,i] = dln.interp(swave1,sflux1,wave1[ind],extrapolate=False,assume_sorted=False)
                        oerr[ind,i] = dln.interp(swave1,serr1,wave1[ind],extrapolate=False,assume_sorted=False,kind='linear')
                        osigma[ind,i] = dln.interp(swave1,ssigma1,wave1[ind],extrapolate=False,assume_sorted=False)
                        mask_interp = dln.interp(swave1,smask1,wave1[ind],extrapolate=False,assume_sorted=False)                    
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

        return ospec
                

    def copy(self):
        """ Create a new copy."""
        new = Spec1D(self.flux.copy(),err=self.err.copy(),wave=self.wave.copy(),mask=self.mask.copy(),
                     lsfpars=self.lsf.pars.copy(),lsftype=self.lsf.lsftype,lsfxtype=self.lsf.xtype,
                     lsfsigma=self.lsf.sigma,instrument=self.instrument,filename=self.filename)
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
                return 0.0
            obs = EarthLocation.of_site(self.observatory)
            ra = self.head.get('ra')
            dec = self.head.get('dec')  
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
        
