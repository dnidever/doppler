.. fraunhofer documentation master file, created by
   sphinx-quickstart on Tue Feb 16 13:03:42 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

**********
Fraunhofer
**********

Introduction
============
|Fraunhofer| [#f1]_ is a generic stellar abundance determination package that fits observed spectrum with synthetic spectral (from the Synspec package).
|Fraunhofer| uses the `Doppler <https://github.com/dnidever/doppler>`_ package to get an initial estimate of the main stellar parameters (Teff, logg,
[Fe/H]) and radial velocity.  It then performs a least-squares fit of the data by searching through parameter space and generated a new Synspec synthetic
spectrum (using the `synple <https://github.com/callendeprieto/synpl>`_ python wrapper package) at each position.

.. toctree::
   :maxdepth: 1

   install
   

Description
===========
|Fraunhofer| has two main modes of operation: 1) a multi-step, iterative fit of the spectrum; or 2) fitting a set of input parameters directly using least-squares.
The first option is the default and is the easiest and most automatic way to fit a spectrum.  The second option can be used for more hands-on situations.

The default, multi-step approach:

1. Fit Teff/logg/[Fe/H]/RV using Doppler
2. Fit Teff/logg/[Fe/H]/RV + vsini with Doppler model
3. Fit stellar parameters (Teff/logg/[Fe/H]/[alpha/H]), RV and broadening (Vrot/Vmicro)
4. Fit each element one at a time holding everything else fixed
5. Fit everything simultaneously

|Fraunhofer| can be called from python directly or the command-line script `hofer` can be used.


Examples
========

.. toctree::
    :maxdepth: 1

    examples


hofer
=====
Here are the various input arguments for command-line script `hofer`::

  usage: hofer [-h] [-e ELEM] [-f FPARS] [-i INIT] [--outfile OUTFILE]
               [--figfile FIGFILE] [-d OUTDIR] [-l LIST] [-p] [--vmicro]
               [--vsini] [-r READER] [-v [VERBOSE]]
               files [files ...]

  Run Fraunhofer fitting on spectra

  positional arguments:
    files                 Spectrum FITS files

  optional arguments:
    -h, --help            show this help message and exit
    -e ELEM, --elem ELEM  List of elements to fit
    -f FPARS, --fpars FPARS
                          List of parameteres to fit
    -i INIT, --init INIT  Initial parameters to use
    -o, --outfile OUTFILE Output filename
    --figfile FIGFILE     Figure filename
    -d OUTDIR, --outdir OUTDIR
                          Output directory
    -l LIST, --list LIST  Input list of FITS files
    -p, --plot            Save the plots
    --vmicro              Fit vmicro
    --vsini               Fit vsini
    -r READER, --reader READER
                          The spectral reader to use
    -v [VERBOSE], --verbose [VERBOSE]
                          Verbosity level (0, 1, 2)

.. rubric:: Footnotes

.. [#f1] For `Joseph von Fraunhofer <https://en.wikipedia.org/wiki/Joseph_von_Fraunhofer>`_ who was a German physicist and the first one to systematically study the absorption lines in the Sun's spectrum.
