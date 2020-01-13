__all__ = ["spec1d","lsf","rv","cannon","utils","reader"]

from .spec1d import Spec1D
from . import (rv, cannon)

# Add custom readers here:
from . import reader
# >>>from mymodule import myreader
# >>>reader._readers['myreader'] = myreader
# You can also do this in your own code.

def read(filename=None,format=None):
    return reader.read(filename=filename,format=None)
