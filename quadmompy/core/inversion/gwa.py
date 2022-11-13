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
This module contains the Golub-Welsch inversion algorithm and derived algorithms.

"""
import numpy as np
from quadmompy.core.inversion.basic import MomentInversion
from scipy import linalg as la
from quadmompy.core.hankel import HankelMatrix


class GolubWelsch(MomentInversion):
    """
    Golub-Welsch moment inversion algorithm.

    Compute Gaussian quadrature given a set of moments using the algorithm proposed by
    Golub and Welsch [:cite:label:`Golub_1969`], implemented as described in Ref. [:cite:label:`John_2012`].

    Parameters
    ----------
    m_2n : float, optional
        Value for the bottom right element of the Hankel moment matrix corresponding to the moment of 2nth order when 2n moments are given (for odd numbers it does not matter). It is only necessary to complete the matrix and make it positive definite for the Cholesky decomposition not to give an error. Otherwise, it does not affect the results. A very large value should be safe. The default value is `1e200`.
    kwargs :
        See base class `MomentInversion`.


    Attributes
    ----------
    R : array
        Array storing the upper right triangular matrix resulting from the Cholesky decomposition of a Hankel moment matrix. It has one additional column for the algorithm to work with both even and odd number of moments.
    m_2n : float
        Value for the bottom right element of the Hankel moment matrix corresponding to the moment of 2nth order when 2n moments are given (for odd numbers it does not matter). It is only necessary to complete the matrix and make it positive definite for the Cholesky decomposition not to give an error. Otherwise, it does not affect the results. A very large value should be safe.

    Notes
    -----
    Several implementations were tested including direct operations on arrays using NumPy-functions. With the usually small number of moments (~10), this implementation proved to be the fastest.

    References
    ----------
        +-----------------+-----------------------+
        | [Golub_1969]    | :cite:`Golub_1969`    |
        +-----------------+-----------------------+
        | [John_2012]     | :cite:`John_2012`     |
        +-----------------+-----------------------+

    """
    def __init__(self, m_2n=1e200, **kwargs):
        super().__init__(**kwargs)
        self.R = None
        self.m_2n = m_2n

    def _compute_rc(self, moments, n, iodd, alpha, beta):
        # TODO: This might be inefficient and should be improved for cases
        # where this function is called repeatedly with the same moments (e.g. EQMOM root search)
        if iodd == 0:
            H = HankelMatrix.from_moments(np.append(moments, self.m_2n))
        else:
            H = HankelMatrix.from_moments(moments)

        # Compute Cholesky decomposition
        # TypeError will occur the first time this function is called as self.R = None
        # IndexError occurs if a larger number of moments is passed than during the first call (should usually not happen often).
        try:
            self.R[:H.shape[0],:H.shape[1]] = H.chol()
            self.R[:,n+1] = 0.
        except (TypeError, IndexError):
            self.R = np.zeros((H.shape[0], H.shape[1] + 1))
            self.R[:H.shape[0],:H.shape[1]] = H.chol()

        # For the usually small matrices, this turned out to be the fastest way (instead of using numpy diags).
        j = 0
        a = 1./self.R[j,j]
        b = self.R[j,j+1]*a
        alpha[j] = b
        for j in range(1,n):
            alpha[j] = -b
            rjj = self.R[j,j]
            beta[j] = rjj*a

            # TODO: when a quadrature is to be computed this is quite inefficient as the
            # square root is taken later to assemble the Jacobi matrix
            beta[j] *= beta[j]

            a = 1./self.R[j,j]
            b = self.R[j,j+1]*a
            alpha[j] += b


# TODO:
#class GolubWelschAdaptive(GolubWelsch):
