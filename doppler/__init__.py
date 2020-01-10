__all__ = ["spec1d","rv","cannon","utils"]

#from .rv import SpecID
from .spec1d import Spec1D
from . import (rv, cannon)

def read(filename=None):
    return rv.rdspec(filename=filename)
