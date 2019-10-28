#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup, find_packages

setup(name='radvel',
      version='1.0',
      description='Generic Radial Velocity Software',
      author='David Nidever',
      author_email='dnidever@montana.edu',
      url='https://github.com/dnidever/radvel',
#      packages=['radvel'],
      packages=find_packages(exclude=["tests"]),
#      scripts=['bin/job_daemon'],
      requires=['numpy','astropy','scipy','thecannon','dlnpyutils']
)
