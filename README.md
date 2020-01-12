# Doppler
The "Doppler" general-purpose stellar radial velocity determination software.


This package has a bunch of small functions that I find useful while working in python.
Most of the functions are in a module called "dlnpyutils".

# Installation

Until I get this in pypi the best way is the usual git clone and setup install:

```
git clone https://github.com/dnidever/doppler.git
cd doppler
python setup.py install
```

# How it works

Doppler uses Cannon models to create stellar spectrum of various stellar types (e.g, effective temperature, surface gravity and metallicity)


# Using the package

Import the package and load your spectrum
```python
import doppler
spec = doppler.read('spectrum.fits')
```

# Adding a custom reader




# Modules and useful functions

There are 4 main modules:
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


# Future improvements
- Add white dwarf Cannon models.
- Modify to fit in the specutils and astropy framework.
- Add jointfit() function to the rv module that will allow fitting of multiple spectra of the same star at one time.