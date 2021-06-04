********
Examples
********


Running hofer
=============
The simplest way to run |Fraunhofer| is with the command-line script ``hofer``.  The only required argument is the name of a spectrum fits file.

.. code-block:: bash

    hofer spectrum.fits

By default, |hofer| doesn't print anything to the screen.  So let's set the ``--verbose`` or ``-v`` parameter.  And, for starters, let's not fit
any elements, just the main stellar parameters.  We can do that by setting ``--elem none``.

.. code-block:: bash
		
    hofer spectrum.fits -v --elem none

    2021-02-16 12:34:54,342 [INFO ]  Start: 2021-02-16 12:34:54
    2021-02-16 12:34:54,342 [INFO ]   
    2021-02-16 12:34:54,440 [INFO ]  spectrum.fits  S/N= 461.7 
    2021-02-16 12:34:54,440 [INFO ]   
    2021-02-16 12:34:54,471 [INFO ]  Step 1: Running Doppler
    2021-02-16 12:34:58,831 [INFO ]  Teff = 5184.450195
    2021-02-16 12:34:58,831 [INFO ]  logg = 3.786091
    2021-02-16 12:34:58,831 [INFO ]  [Fe/H] = -0.208361
    2021-02-16 12:34:58,831 [INFO ]  Vrel = 6.845586
    2021-02-16 12:34:58,831 [INFO ]  chisq = 9.550253
    2021-02-16 12:34:58,831 [INFO ]  dt = 4.360523 sec.
    2021-02-16 12:34:58,831 [INFO ]   
    2021-02-16 12:34:58,831 [INFO ]  Step 2: Fitting vsini with Doppler model
    2021-02-16 12:35:03,943 [INFO ]  Teff = 5297.027888
    2021-02-16 12:35:03,943 [INFO ]  logg = 3.766628
    2021-02-16 12:35:03,943 [INFO ]  [Fe/H] = -0.310375
    2021-02-16 12:35:03,943 [INFO ]  Vrel = 6.609809
    2021-02-16 12:35:03,943 [INFO ]  Vsini = 2.000000
    2021-02-16 12:35:03,943 [INFO ]  chisq = 10.011922
    2021-02-16 12:35:03,943 [INFO ]  dt = 5.111703 sec.
    2021-02-16 12:35:03,943 [INFO ]  Doppler Vrot=0 chisq is better
    2021-02-16 12:35:03,943 [INFO ]   
    2021-02-16 12:35:03,943 [INFO ]  Step 3: Fitting stellar parameters, RV and broadening
    2021-02-16 12:35:03,949 [INFO ]  Fitting: TEFF, LOGG, FE_H, ALPHA_H, RV
    2021-02-16 12:35:17,991 [INFO ]  (5184.4501953125, 3.786090612411499, -0.20836065709590912, -0.20836065709590912, 6.84558629989624)
    2021-02-16 12:36:42,986 [INFO ]  (5085.9318432369655, 3.4715315236966697, -0.19150387314451214, -0.29746932154148664, 6.821367695986499)
    2021-02-16 12:38:07,597 [INFO ]  (5073.10096809831, 3.499507211290511, -0.18083789814981632, -0.2995638783153608, 6.821733060024638)
    2021-02-16 12:39:32,502 [INFO ]  (5112.943118638461, 3.57806216100567, -0.1605418430593645, -0.30291815792257754, 6.818974111722123)
    2021-02-16 12:40:58,400 [INFO ]  (5091.504527205664, 3.5653917571881677, -0.17586455191872563, -0.31138964919010087, 6.817483213624658)
    2021-02-16 12:42:52,612 [INFO ]  (5092.392473200676, 3.563999093409344, -0.172540293018652, -0.310424090980094, 6.816924523433014)
    2021-02-16 12:44:04,296 [INFO ]  Best values:
    2021-02-16 12:44:04,296 [INFO ]  TEFF = 5092.392473
    2021-02-16 12:44:04,296 [INFO ]  LOGG = 3.563999
    2021-02-16 12:44:04,296 [INFO ]  FE_H = -0.172540
    2021-02-16 12:44:04,296 [INFO ]  ALPHA_H = -0.310424
    2021-02-16 12:44:04,296 [INFO ]  RV = 6.816925
    2021-02-16 12:44:18,455 [INFO ]  chisq =  8.28
    2021-02-16 12:44:18,455 [INFO ]  nfev = 39
    2021-02-16 12:44:18,455 [INFO ]  dt = 554.511738 sec.
    2021-02-16 12:44:18,456 [INFO ]  Tweaking continuum using best-fit synthetic model
    2021-02-16 12:44:18,463 [INFO ]   
    2021-02-16 12:44:18,463 [INFO ]  Step 4: Fitting each element separately
    2021-02-16 12:44:18,463 [INFO ]  No elements to fit
    2021-02-16 12:44:18,463 [INFO ]  dt = 564.022548 sec.
    2021-02-16 12:44:18,463 [INFO ]  Writing output to spectrum_fraunhofer.fits
    2021-02-16 12:44:18,591 [INFO ]   
    2021-02-16 12:44:18,591 [INFO ]  End: 2021-02-16 12:44:18
    2021-02-16 12:44:18,591 [INFO ]  elapsed: 564.2 sec.

Steps 4 and 5 are skpped when no elements are being fit.  Just fitting the main stellar parameters and [alpha/H] takes almost 10 minutes.
It takes almost two hours to fit the default 18 elements ['C','N','O','NA','MG','AL','SI','K','CA','TI','V','CR','MN','CO','NI','CU','CE','ND'].

Fitting specific parameters
---------------------------
To fit a specific set of parameters input the comma-separated list using the ``--fpars`` option.  You can also specify initial values for
these parameters (or for parameters not being fit) using the ``--init`` option.  The initial estimates input must be a comma-separated list
of key-value pair (using : or = between the key and value).  Note, that all element names just end with ``_h`` (i.e. [X/H]), for example, ``fe_h`` or ``ca_h``.

This will fit Teff, logg, [Fe/H], and RV.

.. code-block:: bash

    hofer spectrum.fits -v --fpars teff,logg,fe_h,rv

We can give it some initial estimates and fix certain abundances.

.. code-block:: bash

    hofer spectrum.fits -v --fpars teff,logg,fe_h,rv --init teff:5100,logg:3.0,rv:100.0,ca_h:-0.5


Running Fraunhofer from python
==============================
The main |Fraunhofer| module for spectral fitting is ``specfit``.  The ``fit()`` function performs the default multi-step, iterative fitting,
while ``fit_lsq()`` performs least-squares fitting for a specific set of parameters.

This will run the multi-step approach:

    >>> import doppler
    >>> from fraunhofer import specfit
    >>> # Read the spectrum
    >>> spec = doppler.read(filename)
    >>> out, model = specfit.fit(spec,verbose=1)
	

To fit a specific set of parameters, you'll want to input a ``params`` dictionary of the initial and fixed values, and the ``fitparams`` list
of parameters to fit.

    >>> import doppler
    >>> from fraunhofer import specfit
    >>> # Read the spectrum
    >>> spec = doppler.read(filename)
    >>> params = {'teff':5100,'logg':3.0,'fe_h':-1.0,'rv':-50.0,'ca_h':-1.0}
    >>> fitparams=['teff','logg','fe_h','rv','ca_h']
    >>> out, model = specfit.fit_lsq(spec,params,fitparams,verbose=1)
	



    
