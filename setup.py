#!/usr/bin/env python

from distutils.core import setup

setup(name='radvel',
      version='1.0',
      description='Generic Radial Velocity Software',
      author='David Nidever',
      author_email='dnidever@montana.edu',
      url='https://github.com/dnidever/radvel',
      packages=['radvel'],
#      scripts=['bin/job_daemon'],
      requires=['numpy','astropy','scipy','thecannon','dlnpyutils']
)
