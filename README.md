# Doppler
The "Doppler" general-purpose stellar radial velocity determination software.

Determine the radial velocity (RV) and stellar parameters for a spectrum of any wavelength (3000-18000A) and resolution (R<20,000 at the blue end and 120,000 at the red end) with minimal setup.

# Installation

Until I get this in pypi the best way is the usual git clone and setup install:

```
git clone https://github.com/dnidever/doppler.git
cd doppler
python setup.py install
```

# How it works

Doppler uses Cannon models to create stellar spectra of various stellar types (e.g, effective temperature, surface gravity and metallicity).
The Cannon is machine-learning software generally trained on observed stellar spectrum.  I have trained it instead on a large grid of synthetic
stellar spectra and use it as a quick spectral interpolator.  The RV and stellar parameter fitting are done iteratively.  First, the stellar
parameter space is sparsely sampled and the synthetic spectra cross-correlated with the observed spectrum.  Then the best-fitting RV is used
to shift the spectrum to it's rest wavelength scale and the Cannon is used to determine the best stellar parameters.  The RV is fit again with
this new model and then Cannon is used to update the best stellar parameters.  At the end, MCMC (with "emcee") can be run to determine more
accurate uncertainties in the output parameters.

# Quickstart

Import the package and load your spectrum
```python
import doppler
spec = doppler.read('spectrum.fits')
```

Now fit the spectrum:
```python
out, model = doppler.rv.fit(spec)
```


# The Spec1D Object

The Doppler package represents spectra as Spec1D objects.  The 1D indicates that the spectra are 1-dimensional in the sense that
they are flux versus wavelength.  The main properties of a Spec1D object are:
- flux (required): Flux array with dimensions of [Npix] or if multiple orders [Npix,Norder].
- err (optional, but highly recommended): Flux uncertainty array (same shape as flux).
- wave (optional, but highly recommended): Wavelength array (same shape as flux).
- mask: Boolean bad pixel mask for each pixel (same shape as flux).
- bitmask (optional): Bit-mask giving specific information for each pixel.
- lsf: Line Spread Function (LSF) object for this spectrum.
- filename (optional): Name of spectrum file.
- instrument (optional): Instrument on which the spectrum was observed.

Some important Spec1D methods are:
- normalize(): Normalize the spectrum.
- wave2pix(): Convert wavelengths to pixels (takes ```order``` keyword).
- pix2wave(): Convert pixels to wavelengths (takes ```order``` keyword).
- interp(): Interpolate the spectrum onto a new wavelength scale.
- copy(): Make a copy of this spectrum.
- barycorr(): Compute the barycentric correction for this spectrum.

## The Line Spread Function (LSF)

It's important to have some information about the spectrum's Line Spectra Function (LSF) or the width and shape of a spectral line
as a function of wavelength (or pixel).  This is necessary to properly convolve the Cannon model spectrum to the observed spectrum.

Doppler has two types of LSF models: Gaussian or Gauss-Hermite.  The Gauss-Hermite LSF is specifically for APOGEE spectra.  The
Gaussian LSF type should suffice for most other spectra.

If you don't have the LSF information, you can derive it by using comparison or arc-lamp spectra for the same instrumental setup.
Fit Gaussians to all of the mission lines and then fit the Gaussian sigma as a function of pixel or wavelength.  Make sure that the
sigma units and X units are the same (e.g., wavelength in Angstroms or pixels).

The LSF can be easily computed for the entire spectrum by using the ```array()``` method:
```python
lsf = spec.lsf.array()
```
The output will be a 2D array [Npix,Nlsf] if there is only one order, otherwise 3D [Npix,Nlsf,Norder].

You can obtain the LSF for specific pixels or wavelengths with the ```anyarray()``` method:
```python
lsf = spec.lsf.anyarray([100,200],xtype='pixels')
```

The output will be [Npix,Nlsf].

The default of ```anyarray()``` is to put the LSF on the original wavelength scale.  If you want your own new wavelength scale,
then set ```original=True```.

You can use ```utils.convolve_sparse()``` to convolve a flux array with an LSF.
```python
convflux = utils.convolve_sparse(flux,lsf)
```


## The Mask

The Spec1D object uses an internal boolean bad pixel mask called ```mask```, where good pixels have ```False``` and bad pixels ```True``` values.
Normally spectra come with bitmasks which give specific information for each pixel, such as "bad pixel", "saturated", "bright sky line", etc.
But this information is survey or instrument-specific.  Therefore, it is best for each reader to specify the good and bad pixels in ```mask```.
It is also good to save the bitmask information (if it exists) in ```bitmask```.


## Spectral Orders

The Doppler Spec1D objects can represent spectra with multiple "orders".  For example, an APOGEE visit spectrum has components from three
detectors are represented as a [4096,3] array.  These spectra are non-overlapping, but that might not necessarily always be the case.
DESI spectra have components from three spectrograph arms and their spectra overlap.  Doppler handles both of these situations.

Doppler can handle multi-order spectra, but the order dimension must always be the last/trailing one.  For example, flux dimensions
of [4096,3] is okay, but [3,4096] is not. 

Most Doppler functions and methods have a ```order=X``` keyword if a specific order is desired.


## Vacuum or Air Wavelengths

In the past it was the norm to use "air" wavelengths (standard temperature and pressure), but more recently "vacuum" wavelengths are
becoming more common.  Doppler can handle wavelengths in both vacuum or air and will properly convert between them as long as it knows
what the observed spectrum uses.  Make sure to set the ```wavevac``` property is properly set.

## Normalization

The Spec1D objects have a normalize() method, but you may choose to normalize your spectrum in a different way.   Just make sure to:
- normalize the ERR array as well the FLUX
- save the continuum used in CONT, e.g., ```spec.cont = cont```
- and indicate that the spectrum is normalized by setting ```spec.normalized = True```

Then you can use the rest of the Doppler functions just as if the normalize() method was used.

# Creating and adding a custom reader

The current spectral readers are geared towards SDSS-style spectra, but it's easy to create a new custom spectral reader and add it to the list
of readers.  Check the various functions in reader.py to see examples of how to do it.

#### Here are the basics:
```python
from doppler.spec1d import Spec1D
def myreader(filename):
    # Load the flux and wavelength (maybe from header keywords)
    spec = Spec1D(flux,wave=wave)
    return spec
```
If you are going to add LSF information:
```python
from doppler.spec1d import Spec1D
def myreader(filename):
    # Load the flux and wavelength (maybe from header keywords)
    # Load the 
    spec = Spec1D(flux,wave=wave,lsfcoef=lsfcoef,lsftype='Gaussian',lsfxtype='Wave')
    return spec
```
If you know the Gaussian sigma as a function of Wavelength or Pixels, use LSFCOEF.  NOTE, if your lsfxtype is 'Wave',
then it is assumed that the resulting sigmas are also in wavelength units (Angstroms).
If you want to input an array of sigmas instead, then use LSFSIGMA.

There is also a 'Gauss-Hermite' LSF type, but currently this is only used for APOGEE spectra.

It's useful to check at the beginning of your reader that the file/spectrum is of the right type.  If it's not then just return ```None```.

Doppler can handle wavelengths in both vacuum or air and will properly convert between them as long as it knows what the
observed spectrum uses.  Make sure to set ```wavevac = True``` for vacuum wavelengths or ```False``` for air wavelengths.

#### To add your new custom reader to the list of readers:
```
from doppler import reader
from mymodule import myreader
reader._readers['myreader'] = myreader
```

Now your reader should work with ```doppler.read()```.


# Modules and useful functions

There are 6 main modules:
- rv: This module has RV-related functions:
  - fit(): Fit a Cannon model to a spectrum
  - ccorrelate(): cross-correlate two flux arrays.
  - normspec(): Normalize a spectrum.  Spec1D objects use this for their normalize() method.
- cannon: This module has functions for working with Cannon models:
  - model_spectrum(): Generate a stellar spectrum for specific stellar parameters and radial velocity from a Cannon model.
  - prepare_cannon_model(): Prepare a Cannon model (or list of them) for a specific spectrum.
- spec1d: Contains the Spec1D class and methods.
- lsf: Contains the Lsf classes and methods.
- utils: Various utlity functions.
- reader: Contains the spectrum readers.

# Examples

Read in the example spectrum
```python
datadir = doppler.utils.datadir()
spec = doppler.read(datadir+'spec-3586-55181-0500.fits')
```

Print out it's properties:
```python
spec
<class 'doppler.spec1d.Spec1D'>
BOSS spectrum
File = /Users/nidever/projects/doppler/doppler/data/spec-3586-55181-0500.fits
S/N =   54.98
Flux = [ 53.945076 118.17894   80.38118  ...  35.32331   42.468388  39.03381 ]
Err = [11.138297   7.801231   8.071927  ...  5.9317183  6.7650976  6.505177 ]
Wave = [ 3561.2297  3562.0508  3562.8704 ... 10322.862  10325.231  10327.612 ]
```

Now fit the spectrum:
```python
out,model = doppler.rv.fit(spec)
```

The output will be a table with the final results and the best-fitting model spectrum.

# Future improvements
- Add white dwarf Cannon models.
- Modify to fit in the specutils and astropy framework.
- Add jointfit() function to the rv module that will allow fitting of multiple spectra of the same star at one time.
- Add unit support.
- Move docs to readthedocs.