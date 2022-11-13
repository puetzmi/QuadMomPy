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
Module for the correction of unrealizable moment sequences of different types.

"""
import numpy as np
from quadmompy.moments.transform import rc2mom


# TODO: Correct Hamburger moment sequence (particularly important for GaG-CQMOM)
def correct_hamburger(mom, rc=None, inv=None):
    r"""
    Correct unrealizable Hamburger moment sequence by modifying first negative recurrence coefficient :math:`\beta` and computing the corresponding moments.

    Parameters
    ----------
    mom : array
        Moment sequence to be corrected.
    rc : tuple, optional
        Tuple containing the two sets of recurrence coefficients of orthogonal polynomials corresponding to the given moment set.
    inv : MomentInversion, optional
        Basic moment inversion algorithm. Must be provided if recurrence coefficients are not given.

    Returns
    -------
    mom_corr : array
        Modified realizable moment set.

    Raises
    ------
    ValueError :
        If neither a basic inversion algorithm nor the recurrence coefficients are provided.

    """
    pass
