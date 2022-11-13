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
QuadMomPy Moments
-----------------
Useful operations and functions concerning moments, e.g. moments of special distributions and transformations of moments, canonical moments, orthogonal polynomials and related quantities.

"""
from . import transform
from . import special

__all__ = [ \
            "transform", \
            "special", \
          ]
