.. doppler documentation master file, created by
   sphinx-quickstart on Tue Feb 16 13:03:42 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

**********
Doppler
**********

Introduction
============
|Doppler| [#f1]_ is a general-purpose stellar radial velocity determination software.  It uses a forward-modeling approach, convolving a
model spectrum to the resolution or Line Spread Function (LSF) of the observed spectrum.  |Doppler| can be used with a high-resolution
model of the `The Cannon <https://github.com/andycasey/AnniesLasso>`_ (`Casey et al. 2016 <https://ui.adsabs.harvard.edu/abs/2016arXiv160303040C/abstract>`_)
and also of `The Payne <https://github.com/tingyuansen/The_Payne>`_ (`Ting et al. 2019 <https://ui.adsabs.harvard.edu/abs/2019ApJ...879...69T/abstract>`_),
both machine-learning approaches to modeling stellar spectra.
Doppler can determine the radial velocity (RV) and stellar parameters
for a spectrum of any wavelength (3000-18000A) and resolution (R<20,000 at the blue end and 120,000 at the red end) with minimal setup.

The current set of three Cannon models cover temperatures of 3,500K to 60,000K with 3-parameter (Teff, logg, [Fe/H]) and radial velocity.
The current Payne model covers temperatures of 3,500K to 6,000K with 33 labels (Teff, logg, Vmicro, [C/H], [N/H], [O/H], [Na/H], [Mg/H], [Al/H],
[Si/H], [P/H], [S/H], [K/H], [Ca/H], [Ti/H], [V/H], [Cr/H], [Mn/H], [Fe/H], [Co/H], [Ni/H], [Cu/H], [Ni/H], [Cu/H], [Ge/H], [Ce/H],
[Nd/H], [Ba/H], [Eu/H], [La/H], [Y/H], [Sc/H], [Zr/H], [Pr/H], [Yb/H]) as well as radial velocity, rotational velocity and macrotubulence.

|Doppler| also has the ability to simultaneously fit ("jointfit") multiple spectra of a star, with a single set of stellar parameters and elemental
abundances and separate radial velocities for each spectrum.

.. toctree::
   :maxdepth: 1

   install
   gettingstarted
   examples
   modules
	      

Description
===========
|Doppler| fits spectra using a multi-step approach to zero-in on the best solution.

The default, multi-step approach using the Cannon is:

1. Get initial RV using cross-correlation with rough sampling of Teff/logg/[Fe/H] parameter space.
2. Get improved Cannon stellar parameters using initial RV.
3. Improved RV using better Cannon template.
4. Improved Cannon stellar parameters.
5. Full least-squares fitting of all stellar parameters and RV.
6. Run fine-grid in RV using template from previous step
7. Run MCMC (if requested).

The approach with the Payne is:

1. Get initial RV using cross-correlation with rough sampling of Teff/logg/[Fe/H]/[alpha/Fe] parameter space.
2. Least-squares fitting of all desired Payne labels and RV, using best-fit of previous step as initial guess.
3. Run fine-grid in RV using best-fit template from previous step.
4. Run MCMC (if requested).

When jointfit is used,

1. Run regular |Doppler| fit on each spectrum separately.
2. Find weighted mean of all labels and Vhelio.
3. Fit all spectra simultaneously determining one set of labels and a separate RV for each spectrum.

   
|Doppler| can be called from python directly or the command-line script ``doppler`` can be used.


Examples
========

.. toctree::
    :maxdepth: 1

    examples
    gettingstarted


Doppler
=======
Here are the various input arguments for command-line script ``doppler``::

  usage: doppler [-h] [--outfile OUTFILE] [--payne] [--fitpars FITPARS]
                 [--fixpars FIXPARS] [--figfile FIGFILE] [-d OUTDIR] [-l] [-j]
                 [--snrcut SNRCUT] [-p] [-c] [-m] [-r READER] [-v]
                 [-nth NTHREADS] [--notweak] [--tpoly] [--tpolyorder TPOLYORDER]
                 files [files ...]

  Run Doppler fitting on spectra

  positional arguments:
    files                 Spectrum FITS files or list

  optional arguments:
    -h, --help            show this help message and exit
    --outfile OUTFILE     Output filename
    --payne               Fit a Payne model
    --fitpars FITPARS     Payne labels to fit (e.g. TEFF,LOGG,FE_H
    --fixpars FIXPARS     Payne labels to hold fixed (e.g. TEFF:5500,LOGG:2.3
    --figfile FIGFILE     Figure filename
    -d OUTDIR, --outdir OUTDIR
                          Output directory
    -l, --list            Input is a list of FITS files
    -j, --joint           Joint fit all the spectra
    --snrcut SNRCUT       S/N threshold to fit spectrum separately
    -p, --plot            Save the plots
    -c, --corner          Make corner plot with MCMC results
    -m, --mcmc            Run MCMC when fitting spectra individually
    -r READER, --reader READER
                          The spectral reader to use
    -v, --verbose         Verbose output
    -nth NTHREADS, --nthreads NTHREADS
                          Number of threads to use
    --notweak             Do not tweak the continuum using the model
    --tpoly               Use low-order polynomial for tweaking
    --tpolyorder TPOLYORDER
                          Polynomial order to use for tweaking

*****
Index
*****

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`			  
			  
.. rubric:: Footnotes

.. [#f1] For `Christian Doppler <https://en.wikipedia.org/wiki/Christian_Doppler>`_ who was an Austrian physicist who discovered the `Doppler effect <https://en.wikipedia.org/wiki/Doppler_effect>`_ which is the change in frequency of a wave due to the relative speed of the source and observer.
