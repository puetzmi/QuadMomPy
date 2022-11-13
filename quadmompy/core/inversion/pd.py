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
This module contains the product-difference inversion algorithm and derived algorithms.

"""
import numpy as np
from quadmompy.core.inversion.basic import MomentInversion

class ProductDifference(MomentInversion):
    """
    Product-difference moment inversion algorithm [:cite:label:`Gordon_1968`].

    Compute Gaussian quadrature given a set of moments using the product-difference (PD) algorithm [:cite:label:`Gordon_1968`].

    Parameters
    ----------
    kwargs :
        See base class `MomentInversion`.

    Attributes
    ----------
    zeta : array
        Stored continued-fraction coefficients computed during calculation of recursion coefficients.

    Notes
    -----
    Several implementations were tested including NumPy routines. Given only
    a few moments (< 20), this implementation proved to be the most efficient.

    References
    ----------
        +------------------+------------------------+
        | [Gordon_1968]    | :cite:`Gordon_1968`    |
        +------------------+------------------------+

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.zeta = np.array([])

    def _compute_rc(self, moments, n, iodd, alpha, beta):
        size = 2*n + 1 - iodd
        p_mat = np.zeros((size, size))
        p_mat[0,0] = 1.
        p_mat[:-1,1] = moments
        p_mat[::2] *= -1

        for j in range(2, size):
            k = size + 2 - j
            p_mat[:(k-1),j] = p_mat[0,j-1]*p_mat[1:k,j-2] - p_mat[0,j-2]*p_mat[1:k,j-1]

        self.zeta = np.zeros(size - 1)
        self.zeta[1:] = p_mat[0,2:]/p_mat[0,1:-1]/p_mat[0,:-2]

        alpha[:n-iodd] = self.zeta[1::2] + self.zeta[:-1:2]
        beta[1:] = self.zeta[2::2]*self.zeta[1:-1:2]


class ProductDifferenceAdaptive(ProductDifference):
    """
    Product-difference moment inversion algorithm [:cite:label:`Gordon_1968`] (adaptive version).

    Compute Gaussian quadrature given a set of moments using the product-difference (PD) algorithm [:cite:label:`Gordon_1968`]. This adaptive modification uses the same criteria as the adaptive Wheeler algorithm [:cite:label:`Marchisio_2013`] to dynamically reduce the number nodes for increased numerical stability.

    Parameters
    ----------
    rmin : float
        Tolerance for weight-ratio criterion.
    ebs : float
        Tolerance for node-distance criterion.
    kwargs :
        See base class `ProductDifference`.

    Attributes
    ----------
    rmin : float
        Tolerance for weight-ratio criterion.
    ebs : float
        Tolerance for node-distance criterion.

    References
    ----------
        +------------------+------------------------+
        | [Gordon_1968]    | :cite:`Gordon_1968`    |
        +------------------+------------------------+
        | [Marchisio_2013] | :cite:`Marchisio_2013` |
        +------------------+------------------------+

    """
    def __init__(self, rmin=1e-8, eabs=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.rmin = rmin
        self.eabs = eabs

    def moment_inversion(self, moments):
        return self._moment_inversion_ad(moments)
