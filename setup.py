#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup, find_packages

setup(name='Doppler',
      version='1.0.0',
      description='Generic Radial Velocity Software',
      author='David Nidever',
      author_email='dnidever@montana.edu',
      url='https://github.com/dnidever/doppler',
      packages=find_packages(exclude=["tests"]),
      scripts=['bin/dopfit','bin/dopjointfit','bin/doppler'],
      install_requires=['numpy','astropy(>=4.0)','scipy','the-cannon','dlnpyutils(>=1.0.2)','emcee'],
      include_package_data=True,
)
