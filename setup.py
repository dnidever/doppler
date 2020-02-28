#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup, find_packages

setup(name='Doppler',
      version='1.0',
      description='Generic Radial Velocity Software',
      author='David Nidever',
      author_email='dnidever@montana.edu',
      url='https://github.com/dnidever/doppler',
      packages=find_packages(exclude=["tests"]),
      scripts=['bin/dopfit','bin/dopjointfit','bin/doppler'],
      requires=['numpy','astropy(>=4.0)','scipy','thecannon','dlnpyutils'],
      include_package_data=True,
)
