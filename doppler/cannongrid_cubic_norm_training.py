#!/usr/bin/env python

# Imports
import os
import thecannon as tc
from astropy.io import fits
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
import pickle

#tag = '3000_18000_allstars'
#tag = '3000_18000_rgb'
#tag = '3000_18000_cooldwarfs'
tag = '3000_18000_hotstars'
#tag = '3000_18000_hotdwarfs'
#tag = '3000_18000_coolstars'
#tag = '3000_18000_whitedwarfs'


print(tag)

# Import the spectra and labels
normalized_flux = fits.getdata('cannongrid_'+tag+'_synth_data_norm.fits.gz')
labelled_set=Table.read('cannongrid_'+tag+'_synth_pars.fits')

# Add wavelengths
nspec, npix = normalized_flux.shape
wave = np.arange(npix)*0.10+3000.0
#model3.dispersion = wave
normalized_ivar = normalized_flux.copy()*0 + 1e4
vec3 = tc.vectorizer.PolynomialVectorizer(labelled_set.colnames, 3)
model3 = tc.CannonModel(labelled_set, normalized_flux, normalized_ivar, vec3, wave)
model3.regularization = 0   # no regularization for now

# Train the model
nr_theta, nr_s2, nr_metadata = model3.train()  
if os.path.exists('cannongrid_'+tag+'_norm_cubic_model.pkl'): os.remove('cannongrid_'+tag+'_norm_cubic_model.pkl')
model3.write('cannongrid_'+tag+'_norm_cubic_model.pkl')

# Check if there is a continuum model to add
contfile = 'cannongrid_'+tag+'_cont_cubic_model_logflux.pkl'
if os.path.exists(contfile):
    print('Continuum model found.  Adding it')
    f = open(contfile,'rb')
    cont = pickle.load(f)
    f.close()
else:
    cont = None

# This adds my custom attributes
#   model.write() doesn't same them
infile = open('cannongrid_'+tag+'_norm_cubic_model.pkl','rb')
temp = pickle.load(infile)
infile.close()
temp['fwhm'] = 0.001
temp['wavevac'] = False
if cont is not None:
    temp['continuum'] = cont
md = temp['metadata']
ta = md['trained_attributes']
if cont is not None:
    ta = ta+('fwhm','wavevac','continuum')
else:
    ta = ta+('fwhm','wavevac')
md['trained_attributes'] = ta
temp['metadata'] = md
os.remove('cannongrid_'+tag+'_norm_cubic_model.pkl')
outfile = open('cannongrid_'+tag+'_norm_cubic_model.pkl','wb')
pickle.dump(temp,outfile)
outfile.close()

# Test/fit the spectra to get the labels
labels3, cov, meta = model3.test(normalized_flux, normalized_ivar)


# one-to-one comparisons
names = model3.vectorizer.label_names
fig = plt.figure(figsize=(15,10))
for i in range(0, 3):
    plt.subplot(2, 2, i+1)
    plt.plot(labelled_set[names[i]],labels3[:,i],'+')
    plt.plot(labelled_set[names[i]],labelled_set[names[i]])
    plt.xlabel('Input '+names[i])
    plt.ylabel('Recovered '+names[i])
fig.savefig('cannongrid_'+tag+'_comparison.png')    


