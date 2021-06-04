************
Installation
************


Important packages
==================
`Doppler <https://github.com/dnidever/doppler>`_ is a generic radial velocity package and is needed to read spectra, get initial parameters and RVs, and other things.

``synple`` is Carlos Allende Prieto's python wrapper around the `Synspec <http://tlusty.oca.eu/Synspec49/synspec.html>`_ spectral synthesis software.
It includes the Synspec fortran code, linelists, and python code needed.  Currently, `dnidever's fork of synple <https://github.com/dnidever/synple>`_ is needed.  Note, that you need to both compile the fortran code and `python setup.py install` the python code.

Installing Fraunhofer
=====================

Until I get Fraunhofer set up with PyPi, it's best to install it from source using GitHub.

.. code-block:: bash

    git clone https://github.com/dnidever/fraunhofer.git

Then install it:

.. code-block:: bash

    python setup.py install



Dependencies
============

- numpy
- scipy
- astropy
- matplotlib
- `synple <https://github.com/dnidever/synple>`_ (dnidever's fork of Carlos Allende Prieto's `synple <https://github.com/callendeprieto/synpl>`_  package)
- `dlnpyutils <https://github.com/dnidever/dlnpyutils>`_
- `doppler <https://github.com/dnidever/doppler>`_
