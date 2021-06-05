********
Examples
********


Running doppler
===============
The simplest way to run |Doppler| is with the command-line script ``doppler``.  The only required argument is the name of a spectrum fits file.

.. code-block:: bash

    doppler spectrum.fits

This will create an output file called ``spectrum_doppler.fits``.

By default, |Doppler| doesn't print anything to the screen.  So let's set the ``--verbose`` or ``-v`` parameter.

.. code-block:: bash
		
    doppler spectrum.fits -v

    spectrum.fits  S/N=  85.1 
    Fitting Cannon model to spectrum with parameters: Teff, logg, [Fe/H], RV
    Masked 103 outlier or negative pixels
    TEFF    LOGG     FEH    VREL   CCPEAK    CHISQ
    3500.00   4.80  -0.50  -13.67   0.23   7.87
    4000.00   4.80  -0.50  -15.37   0.39   5.41
    5000.00   4.60  -0.50  -16.56   0.50   3.85
    6000.00   4.40  -0.50  -16.77   0.48   5.16
    7500.00   4.00  -0.50  -16.90   0.16   7.65
    15000.00   4.00  -0.50  989.79   0.03   8.32
    25000.00   4.00  -0.50  -999.68   0.05   8.81
    40000.00   4.00  -0.50  -308.33   0.08   8.96
    3500.00   0.50  -0.50  -15.19   0.34  13.01
    4300.00   1.00  -0.50  -16.22   0.45   6.58
    4700.00   2.00  -0.50  -16.55   0.50   4.68
    5200.00   3.00  -0.50  -16.81   0.50   4.45
    Initial RV fit:
    Teff   =   5000.00 K    
    logg   =      4.60      
    [Fe/H] =     -0.50      
    Vrel   =    -16.56 +/-  0.136 km/s 
    Initial Cannon stellar parameters using initial RV
    Teff   =   5074.34 K    
    logg   =      4.86      
    [Fe/H] =      0.10      
    Masked 48 discrepant pixels
    Initial Cannon stellar parameters using initial RV and Tweaking the normalization
    Teff   =   5049.52 K    
    logg   =      4.98      
    [Fe/H] =      0.14      
    Improved RV and Cannon stellar parameters:
    Teff   =   5049.52 K    
    logg   =      4.98      
    [Fe/H] =      0.14      
    Vrel   =    -16.78 +/-  0.117 km/s 
    Least Squares RV and stellar parameters:
    Teff   =   5049.52 K    
    logg   =      4.98      
    [Fe/H] =      0.14      
    Vrel   =    -16.78 km/s 
    chisq =  1.96
    Fine grid best RV = -16.98 km/s
    chisq =  1.96
    Final parameters:
    Teff   =   5049.52 +/-  3.158 K    
    logg   =      4.98 +/-  0.003      
    [Fe/H] =      0.14 +/-  0.001      
    Vhelio = -18.02 +/-  0.01 km/s
    BC = -1.04 km/s
    chisq =  1.96
    dt =  3.40 sec.
    Writing output to spectrum_doppler.fits

You can also have it save a figure of the spectrum and the best-fit model using the ``--plot`` or ``-p`` parameter.

.. code-block:: bash
		
    doppler spectrum.fits -v -p

    spectrum.fits  S/N=  85.1 
    Fitting Cannon model to spectrum with parameters: Teff, logg, [Fe/H], RV
    Masked 103 outlier or negative pixels
    TEFF    LOGG     FEH    VREL   CCPEAK    CHISQ
    3500.00   4.80  -0.50  -13.67   0.23   7.87
    ...
    chisq =  1.96
    Figure saved to spectrum_dopfit.png
    dt =  4.08 sec.
    Writing output to spectrum_doppler.fits

By default, the figure filename is ``spectrum_dopfit.png``, but the figure filename can be set with the
``--figfile`` parameter.  Note that you can set the figure type with the extension.  For example,
to get a PDF just use ``--figfile spectrum_dopfit.pdf``.

Using a Payne Model
-------------------

By default, |Doppler| uses a Cannon model fitting ``Teff, logg, [Fe/H] and RV``.  To use a Payne model instead,
just set the ``--payne`` parameter.

.. code-block:: bash
		
    doppler spectrum.fits -v --payne

    spectrum.fits  S/N=  85.1 
    Fitting Payne model to spectrum with parameters: TEFF, LOGG, FE_H, ALPHA_H, RV
    Masked 103 outlier or negative pixels
      TEFF    LOGG   FEH   ALPHAFE    VREL  CCPEAK  CHISQ
       3500   4.80  -1.50   0.00    -13.86   0.13   8.67
       4000   4.80  -1.50   0.00    -15.52   0.17   8.21
       5000   4.60  -1.50   0.00    -17.53   0.33   7.89
       6000   4.40  -1.50   0.00    -17.57   0.28   8.25
       7500   4.00  -1.50   0.00    -17.97   0.04   8.20
      15000   4.00  -1.50   0.00    994.28   0.02   8.57
      25000   4.00  -1.50   0.00  -1004.90   0.01   8.83
      40000   4.00  -1.50   0.00  -1004.90   0.01   9.08
       3500   0.50  -1.50   0.00    -14.79   0.25   9.16
       4300   1.00  -1.50   0.00    -16.82   0.40   6.60
       4700   2.00  -1.50   0.00    -17.12   0.37   7.29
       5200   3.00  -1.50   0.00    -17.31   0.33   7.77
       3500   4.80  -0.50   0.00    -13.67   0.16   7.96
       4000   4.80  -0.50   0.00    -15.71   0.30   6.23
       5000   4.60  -0.50   0.00    -16.92   0.45   4.19
       6000   4.40  -0.50   0.00    -17.03   0.43   6.11
       7500   4.00  -0.50   0.00    -16.94   0.15   7.54
      15000   4.00  -0.50   0.00    996.77   0.02   8.60
      25000   4.00  -0.50   0.00  -1004.90   0.01   8.85
      40000   4.00  -0.50   0.00  -1004.90   0.01   9.08
       3500   0.50  -0.50   0.00    -15.56   0.30  12.90
       4300   1.00  -0.50   0.00    -16.60   0.43   6.51
       4700   2.00  -0.50   0.00    -16.81   0.46   4.95
       5200   3.00  -0.50   0.00    -16.99   0.45   5.27
       3500   4.80  -1.50   0.30    -13.75   0.11   8.59
       4000   4.80  -1.50   0.30    -16.02   0.20   8.17
       5000   4.60  -1.50   0.30    -17.21   0.32   7.58
       6000   4.40  -1.50   0.30    -17.37   0.28   8.06
       7500   4.00  -1.50   0.30    -17.60   0.05   8.14
      15000   4.00  -1.50   0.30    994.34   0.02   8.57
      25000   4.00  -1.50   0.30  -1004.90   0.01   8.83
      40000   4.00  -1.50   0.30  -1004.90   0.01   9.08
       3500   0.50  -1.50   0.30    -14.96   0.27   8.73
       4300   1.00  -1.50   0.30    -16.75   0.40   6.43
       4700   2.00  -1.50   0.30    -17.03   0.37   7.13
       5200   3.00  -1.50   0.30    -17.19   0.33   7.60
       3500   4.80  -0.50   0.30    -13.92   0.18   7.74
       4000   4.80  -0.50   0.30    -15.63   0.32   5.93
       5000   4.60  -0.50   0.30    -16.78   0.43   3.98
       6000   4.40  -0.50   0.30    -16.96   0.41   5.84
       7500   4.00  -0.50   0.30    -16.88   0.16   7.45
      15000   4.00  -0.50   0.30    997.53   0.02   8.59
      25000   4.00  -0.50   0.30  -1004.90   0.01   8.85
      40000   4.00  -0.50   0.30   -998.83   0.01   9.08
       3500   0.50  -0.50   0.30    -15.62   0.31  12.53
       4300   1.00  -0.50   0.30    -16.51   0.42   6.98
       4700   2.00  -0.50   0.30    -16.74   0.45   4.99
       5200   3.00  -0.50   0.30    -16.93   0.45   5.05
    Initial RV fit:
    Teff   =   5000.00 K    
    logg   =      4.60      
    [Fe/H] =     -0.50      
    [alpha/Fe] =      0.30      
    Vrel   =    -16.78 +/-  0.093 km/s 
    chisq =  3.98
    Least Squares RV and stellar parameters:
    TEFF   =   4828.16      
    LOGG   =      4.68      
    FE_H   =     -0.03      
    ALPHA_H =     -0.23      
    RV     =    -17.05      
    chisq =  2.15
    Fine grid best RV = -17.08 km/s
    chisq =  2.15
    Final parameters:
    TEFF   =   4828.16 +/- 12.276      
    LOGG   =      4.68 +/-  0.016      
    FE_H   =     -0.03 +/-  0.005      
    ALPHA_H =     -0.23 +/-  0.008      
    RV     =    -17.08 +/-  0.045      
    Vhelio = -18.12 +/-  0.04 km/s
    BC = -1.04 km/s
    chisq =  2.15
    dt = 11.10 sec.
    Writing output to spectrum_doppler.fits

By default, |Doppler| fits the parameters ``Teff, logg, [Fe/H], [alpha/H], and RV`` with a Payne model.
But the parameters can be set using the ``--fitpars`` parameter with a comma-separated list of parameter
names (with no spaces).  Note that all abundance names need to end with ``_h`` such as ``fe_h``,
``alpha_h`` and ``c_h``.  To fit the carbon abundance as well, you would do,

.. code-block:: bash
		
    doppler spectrum.fits -v --payne --fitpars teff,logg,fe_h,alpha_h,c_h,rv

    spectrum.fits  S/N=  85.1 
    Fitting Payne model to spectrum with parameters: TEFF, LOGG, FE_H, ALPHA_H, RV
    Masked 103 outlier or negative pixels
      TEFF    LOGG   FEH   ALPHAFE    VREL  CCPEAK  CHISQ
       3500   4.80  -1.50   0.00    -13.86   0.13   8.67
    ...
    Least Squares RV and stellar parameters:
    TEFF   =   4860.69      
    LOGG   =      4.72      
    FE_H   =     -0.03      
    ALPHA_H =     -0.24      
    C_H    =      0.08      
    RV     =    -17.03      
    chisq =  2.13
    Fine grid best RV = -17.06 km/s
    chisq =  2.13
    Final parameters:
    TEFF   =   4860.69 +/- 16.727      
    LOGG   =      4.72 +/-  0.019      
    FE_H   =     -0.03 +/-  0.007      
    ALPHA_H =     -0.24 +/-  0.010      
    C_H    =      0.08 +/-  0.010      
    RV     =    -17.06 +/-  0.045      
    Vhelio = -18.10 +/-  0.05 km/s
    BC = -1.04 km/s
    chisq =  2.13
    dt = 11.00 sec.

The current Payne model has 33 labels (Teff, logg, Vmicro, [C/H], [N/H], [O/H], [Na/H], [Mg/H], [Al/H], [Si/H], [P/H], [S/H],
[K/H], [Ca/H], [Ti/H], [V/H], [Cr/H], [Mn/H], [Fe/H], [Co/H], [Ni/H], [Cu/H], [Ni/H], [Cu/H], [Ge/H], [Ce/H], [Nd/H], [Ba/H],
[Eu/H], [La/H], [Y/H], [Sc/H], [Zr/H], [Pr/H], [Yb/H]).  Besides RV, Vsini and Vmacro can also be fit with a Payne model.

    
Jointly fitting spectra
-----------------------

|Doppler| can fit mulitple spectra of the same star simultaneously.  It will fit a single set of labels and separate
RVs for each spectrum.  To use this option, set the ``--joint`` or ``j`` parameters.  You also need to give the
names of the mulitiple spectrum files.  You can give these on the command list, e.g. ``doppler spectrum1.fits spectrum2.fits -j``,
or you can put the names of the files in an ASCII list file (one per line) and give that to |Doppler|.  To use a list, be sure
to also set the ``--list`` or ``-l`` parameter.

.. code-block:: bash

    doppler speclist.txt -v -j -l

    --- Running Doppler Jointfit on 3 spectra ---

    Loading the spectra
    Spectrum   1:  apVisit-r13-9289-58006-107.fits  S/N= 102.4 
    Spectrum   2:  apVisit-r13-9289-58028-113.fits  S/N=  74.5 
    Spectrum   3:  apVisit-r13-9289-58054-107.fits  S/N=  72.4 

    Jointly fitting Cannon model to 3 spectra with parameters: Teff, logg, [Fe/H] and RV for each spectrum
    Step #1: Fitting the individual spectra
    Fitting spectrum 1
    apVisit-r13-9289-58006-107.fits
    Fitting Cannon model to spectrum with parameters: Teff, logg, [Fe/H], RV
    ...
    Final parameters:
    Teff   =   4658.20 +/-  1.463 K    
    logg   =      2.62 +/-  0.002      
    [Fe/H] =     -0.16 +/-  0.001      
    Vhelio = -82.05 +/-  0.01 km/s
    BC = 14.38 km/s
    chisq =  2.26
    dt =  4.16 sec.
 
    Fitting spectrum 2
    apVisit-r13-9289-58028-113.fits
    Fitting Cannon model to spectrum with parameters: Teff, logg, [Fe/H], RV
    ...
    Final parameters:
    Teff   =   4666.10 +/-  1.498 K    
    logg   =      2.56 +/-  0.002      
    [Fe/H] =     -0.15 +/-  0.001      
    Vhelio = -82.03 +/-  0.01 km/s
    BC =  8.07 km/s
    chisq =  1.88
    dt =  3.37 sec.
    
    Fitting spectrum 3
    apVisit-r13-9289-58054-107.fits
    Fitting Cannon model to spectrum with parameters: Teff, logg, [Fe/H], RV
    ...
    Final parameters:
    Teff   =   4657.61 +/-  1.624 K    
    logg   =      2.59 +/-  0.002      
    [Fe/H] =     -0.16 +/-  0.001      
    Vhelio = -82.10 +/-  0.01 km/s
    BC = -1.02 km/s
    chisq =  1.70
    dt =  3.28 sec.
 
    Step #2: Getting weighted stellar parameters
    Initial weighted parameters are:
    Teff   =   4660.75 K    
    logg   =      2.59      
    [Fe/H] =     -0.16      
    Vrel   =    -82.06 km/s 
 
    Step #3: Fitting all spectra simultaneously
    Parameters:
    Teff   =   4660.97 K    
    logg   =      2.59      
    [Fe/H] =     -0.16      
    Vhelio = -82.05 +/-  0.01 km/s
    Vscatter =   0.017 km/s
    [-82.04650497 -82.03479767 -82.10112727]
 
    Step #4: Tweaking continuum and masking outliers
    Masked 4 discrepant pixels
    Masked 3 discrepant pixels
    Masked 1 discrepant pixels
    
    Step #5: Re-fitting all spectra simultaneously
    Final parameters:
    Teff   =   4661.19 K    
    logg   =      2.59      
    [Fe/H] =     -0.16      
    Vhelio = -82.05 +/-  0.01 km/s
    Vscatter =   0.017 km/s
    [-82.04650497 -82.03479767 -82.10112727]
    dt = 13.05 sec.
    Writing output to apVisit-r13-9289-58006-107_doppler.fits
	
|Doppler| first fits each spectrum separately.  Then it finds weighted stellar parameters and Vhelio.  Finally,
it fits a single set of stellar parameters and an RV for each spectrum to all spectra simultaneously.
    
    
Running |Doppler| from python
=============================

All of the |Doppler| functionality is also available directory from python.

First, import |Doppler| and read in the example spectrum.

.. code-block:: python

	import doppler
	spec = doppler.read('spectrum.fits')

Print out it's properties:

.. code-block:: python

	spec
	
	<class 'doppler.spec1d.Spec1D'>
	APOGEE spectrum
	File = spectrum.fits
	S/N =   85.06
	Dimensions: [2048,3]
	Flux = [[3.6705207e-15 3.8983997e-15 4.8754683e-15]
	 [3.6727666e-15 3.9004448e-15 4.8746174e-15]
	 [3.6702780e-15 3.9016904e-15 4.8737662e-15]
	 ...
	 [4.3078719e-15 4.5482637e-15 5.8845268e-15]
	 [4.3102292e-15 4.5506604e-15 5.8881089e-15]
	 [4.3116611e-15 4.5502703e-15 5.8918320e-15]]
	Err = [[1.5879806e-08 2.8263509e-08 3.4794638e-08]
	 [1.5880362e-08 2.8273581e-08 3.4788567e-08]
	 [1.5881179e-08 2.8282614e-08 3.4782488e-08]
	 ...
	 [2.7037721e-08 3.5628904e-08 3.2013901e-08]
	 [2.7046704e-08 3.5624442e-08 3.2033391e-08]
	 [2.7055689e-08 3.5619223e-08 3.2053357e-08]]
	Wave = [[16953.30940492 16433.71888594 15808.37977415]
	 [16953.09701271 16433.45786229 15808.07341694]
	 [16952.88459753 16433.19681725 15807.76703994]
	 ...
	 [16471.98020599 15856.26367213 15141.55733694]
	 [16471.72234417 15855.96043257 15141.21203183]
	 [16471.46446085 15855.65717311 15140.86670842]]

Now fit the spectrum:

.. code-block:: python
		
	out,model,specm = doppler.rv.fit(spec)

The output will be a table with the final results, the best-fitting model spectrum, and the masked and tweaked
version of the observed spectrum used internall by |Doppler| for the fitting.


To jointly fit spectra, you want to load all of the spectra into a list and then run ``jointfit()``.

.. code-block:: python

	import doppler
	from dlnpyutils import utils as dln
	# Read in the list of files
	files = dln.readlines(.txt')
	speclist = []
	for i in range(len(files)):
	    spec = doppler.read(files[i]
	    speclist.append(spec)
	# Now run jointfit()
	sumstr,final,bmodel,specmlist = doppler.rv.jointfit_payne(speclist)

The outputs are a summary file of the best-fit values (sumstr), final best-fit values for each individual spectrum (final),
list of best-fitting model spectra (bmodel), and the list of the masked and tweaked observed spectra (specmlist).
