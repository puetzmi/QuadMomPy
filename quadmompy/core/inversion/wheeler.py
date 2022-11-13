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
This module contains the Wheeler inversion algorithm and derived algorithms.

"""
import numpy as np
from quadmompy.core.inversion.basic import MomentInversion


class Wheeler(MomentInversion):
    """
    Wheeler moment inversion algorithm.

    Compute Gaussian quadrature given a set of moments using the Wheeler algorithm proposed by Sack and Donovan [:cite:label:`Sack_1971`) and Wheeler [:cite:label:`Wheeler_1974`]. A set of 2N moments is needed to compute an N-node quadrature.

    Parameters
    ----------
    cutoff : float
        Threshold for sigma-matrix. If sigma[k,k] < cutoff, it is considered zero, which means that the moment sequence is located on the moment space boundary and the quadrature nodes are truncated.
    kwargs :
        See base class `MomentInversion`.

    Attributes
    ----------
    sigma : array
        Stored matrix used to compute recursion coefficients of orthogonal polynomials.
    cutoff : float
        Threshold for sigma-matrix. If sigma[k,k] < cutoff, it is considered zero, which means that the moment sequence is located on the moment space boundary and the quadrature nodes are truncated.

    Notes
    -----
    Several implementations were tested including NumPy routines. Given only
    a few moments (< 20), this implementation proved to be the most efficient.

    References
    ----------
        +------------------+------------------------+
        | [Sack_1971]      | :cite:`Gautschi_2004`  |
        +------------------+------------------------+
        | [Wheeler_1974]   | :cite:`Wheeler_1974`   |
        +------------------+------------------------+

    """
    def __init__(self, cutoff=1e-12, **kwargs):
        super().__init__(**kwargs)
        self.cutoff = cutoff
        self.sigma = np.array([])

    def _compute_rc(self, moments, n, iodd, alpha, beta):
        nmom = len(moments)
        self.sigma = np.zeros((n+1, nmom+1))
        self.sigma[1,1:] = moments
        for i in range(2, n+1):
            jmax = 2*n + 2 - i - iodd
            for j in range(i, jmax):
                self.sigma[i,j] = self.sigma[i-1,j+1]-alpha[i-2]*self.sigma[i-1,j]-beta[i-2]*self.sigma[i-2,j]
            if abs(self.sigma[i,i]) < self.cutoff:
                break
            alpha[i-1] = self.sigma[i,i+1]/self.sigma[i,i]-self.sigma[i-1,i]/self.sigma[i-1,i-1]
            beta[i-1] = self.sigma[i,i]/self.sigma[i-1,i-1]


class WheelerAdaptive(Wheeler):
    """
    Wheeler moment inversion algorithm with adaptive reduction of quadrature nodes.

    Compute Gaussian quadrature given a set of moments using the Wheeler algorithm proposed by Sack and Donovan [:cite:label:`Sack_1971`] and Wheeler [:cite:label:`Wheeler_1974`]. A set of 2N moments is needed to compute an N-node quadrature. The adaptive modification of the Wheeler algorithm [:cite:label:`Marchisio_2013`] dynamically reduces the number nodes to increase numerical stability.

    Parameters
    ----------
    rmin : float
        Tolerance for weight-ratio criterion.
    ebs : float
        Tolerance for node-distance criterion.
    kwargs :
        See base class `Wheeler`.

    Attributes
    ----------
    cutoff : float
        Threshold for sigma-matrix. If sigma[k,k] < cutoff, it is considered zero, which means that the moment sequence is located on the moment space boundary and the quadrature nodes are truncated.
    rmin : float
        Tolerance for weight-ratio criterion.
    ebs : float
        Tolerance for node-distance criterion.

    References
    ----------
        +------------------+------------------------+
        | [Sack_1971]      | :cite:`Gautschi_2004`  |
        +------------------+------------------------+
        | [Wheeler_1974]   | :cite:`Wheeler_1974`   |
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
