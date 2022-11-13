"""
QuadMom-qbmm subpackage with all univariate and multivariate quadrature-based
moment methods.

"""
from . import qbmm
from . import multivariate
from . import univariate
from .qbmm import *

__all__ = [ \
            "qbmm", \
            "univariate", \
            "multivariate", \
          ]
