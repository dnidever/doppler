__all__ = ["spec1d","lsf","rv","cannon","utils","reader","payne"]
__version__ = '1.1.9'

from .spec1d import Spec1D
from . import (rv, cannon)

# Add custom readers here:
from . import reader
# >>>from mymodule import myreader
# >>>reader._readers['myreader'] = myreader
# You can also do this in your own code.

# Saving all the models in cannon.models
models = cannon.load_models()
cannon.models = models

def read(filename=None,format=None):
    return reader.read(filename=filename,format=format)

def fit(*args,**kwargs):
    return rv.fit(*args,**kwargs)

def jointfit(*args,**kwargs):
    return rv.jointfit(*args,**kwargs)

def download_data(force=False):
    utils.download_data(force=force)
