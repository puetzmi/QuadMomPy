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
QuadMomPy subpackage for the solution of specific equations, e.g. population balance equations, with QBMMs.

"""
from . import integrate_1d
from . import fpe_1d
from .fpe_1d import *

__all__ = [ \
            "fpe_1d", \
            "integrate_1d", \
          ]
