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
QuadMomPy Core
--------------
QuadMoM-Core subpackage. Includes basic IO-operations, data types and fundamental computations involving moments, Hankel matrices, Gaussian quadratures and orthogonal polynomials required by QBMMs and other subpackages.

"""

from . import inversion
from . import io
from . import quadrature
from . import hankel
from .hankel import *
from . import utils

__all__ = [ \
            "inversion", \
            "io", \
            "quadrature", \
            "hankel", \
            "utils", \
          ]
