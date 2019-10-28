__all__ = ["rv","cannon"]

#from .rv import SpecID
from . import (rv, cannon)

def read(filename=None):
    return rv.rdspec(filename=filename)
