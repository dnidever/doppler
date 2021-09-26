#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup, find_packages

setup(name='Doppler',
      version='1.1.0',
      description='Generic Radial Velocity Software',
      author='David Nidever',
      author_email='dnidever@montana.edu',
      url='https://github.com/dnidever/doppler',
      packages=find_packages(exclude=["tests"]),
      scripts=['bin/doppler'],
      #install_requires=['numpy','astropy(>=4.0)','scipy','dlnpyutils(>=1.0.3)','emcee','corner','thecannon @ git+ssh://git@github.com/andycasey/AnniesLasso@master#egg=thecannon'],
      install_requires=['numpy','astropy(>=4.0)','scipy','dlnpyutils(>=1.0.3)','dill','emcee','corner',
                        'the-cannon @ git+https://github.com/dnidever/AnniesLasso@v1.0.0#egg=the-cannon'],
      #                  'the-cannon @ git+https://github.com/sdss/thecannon@1.0.0#egg=the-cannon'],      
      #                  'the-cannon @ https://github.com/andycasey/AnniesLasso/archive/master.zip'],      
      #install_requires=['numpy','astropy(>=4.0)','scipy','dlnpyutils(>=1.0.3)','emcee','corner','thecannon @ git+http://github.com/andycasey/AnniesLasso@master#egg=thecannon'],      
      #install_requires=['numpy','astropy(>=4.0)','scipy','dlnpyutils(>=1.0.3)','emcee','corner','thecannon'],
      #install_requires=['numpy','astropy(>=4.0)','scipy','the-cannon','dlnpyutils(>=1.0.2)','emcee','corner'],
      #dependency_links=['http://github.com/andycasey/AnniesLasso/tarball/master'],
      #dependency_links=['http://github.com/andycasey/AnniesLasso/tarball/master#egg=thecannon'],
      dependency_links=['http://github.com/dnidever/AnniesLasso/tarball/v1.0.0#egg=the-cannon'],      
      include_package_data=True,
)
