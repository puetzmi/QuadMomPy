"""
QuadMomPy subpackage for the solution of specific equations, e.g. population balance equations, with QBMMs.

"""
from . import integrate_1d
from . import fpe_1d
from .fpe_1d import *

__all__ = [ \
            "fpe_1d", \
            "integrate_1d", \
          ]
