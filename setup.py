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
      install_requires=['numpy','astropy(>=4.0)','scipy','dlnpyutils(>=1.0.3)','dill','emcee','corner',
                        'the-cannon @ git+https://github.com/dnidever/AnniesLasso@v1.0.0#egg=the-cannon'],
      dependency_links=['http://github.com/dnidever/AnniesLasso/tarball/v1.0.0#egg=the-cannon'],      
      include_package_data=True,
)
