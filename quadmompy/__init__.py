#
# Copyright (c) 2022 Michele Puetz.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

"""
*************************
QuadMomPy Python3 library
*************************
Library for all sorts of fun with moments, Gaussian quadratures and orthogonal polynomials for the solution of spatially homogeneous population balance equations.

References
----------
.. bibliography::
    :style: unsrt

"""

from . import core
from .core import *
from . import qbmm
from . import moments
from . import equations


__all__ = [ \
            "core", \
            "qbmm", \
            "moments", \
            "equations"
          ]

__all__.extend(core.__all__)
