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
Hankel matrices, determinants and related functions.

"""
import numpy as np
import scipy.linalg as la


def hankel_matrix(mom, kind='lower', inplace=False, A=None):
    """
    Make Hankel matrix given the moment sequence `mom`. If the number of moments in `mom` is even (`len(mom)==2*n`) then two different moment matrices may be assembled: (1) The lower Hankel matrix containing the moments of order 0 to 2n-2 or (2) the upper Hankel matrix with moments of orders 1 to 2n-1. If the number of moments in `mom` is odd the Hankel matrix is unique.

    Parameters
    ----------
    mom : array_like
        Moment sequence.
    kind : str, optional
        Specification of Hankel matrix type as explained in the description above; may take the values `lower` or `upper`. If len(m) is odd the choice of `kind` does not have any effect.
    inplace : bool, optional
        Indicates if the function operates directly on a given matrix `A`, which must be provided as additional parameter if `inplace==True`. Default is False.
    A : array, optional
        Array to store the Hankel matrix, only necessary if `inplace==True`.

    Returns
    -------
    H : array_like
        Hankel matrix.

    """
    nm = len(mom)
    size = nm//2 + nm % 2
    if inplace:
        H = A
    else:
        H = np.zeros((size, size))
    if kind == 'lower' or nm % 2 == 1:
        k = 0
    elif kind == 'upper':
        k = 1
    else:
        raise ValueError("Invalid value of parameter `kind`, which must take either the value `upper` or `lower`.")
    for i in range(size):
        H[i] = mom[k+i:size+k+i]
    return H


def hankel_det(mom, kind='lower'):
    """
    Compute Hankel determinant of moment matrix given the moment sequence `mom`.

    Parameters
    ----------
    mom : array_like
        Moment sequence.
    kind : str, optional
        Specification of Hankel matrix type, see `hankel_matrix`; may take the values `lower` or `upper`. If `len(m)` is odd the choice of `kind` does not have any effect.

    Returns
    -------
    det : float
        Hankel determinant.

    """
    H = hankel_matrix(mom, kind)
    return np.linalg.det(H)


class HankelMatrix:
    """
    Class for Hankel-type moment matrices. Direct calls to the constructor are discouraged. Instantiation should occur by calling the `from_moments`-method.

    Parameters
    ----------
    data : array
        2D-array representing the Hankel moment matrix.
    check_pd : bool
        Indicates if positive definiteness is to be checked.

    Attributes
    ----------
    _data : array
        2D-array representing the Hankel moment matrix. Direct manipulation leads to inconsistencies and is thus not recommended.
    _det : float
        Determinant of the Hankel-matrix. It is only computed when it is required the first time.
    _chol : array
        2D-array representing the upper right triangular matrix resulting from a Cholesky-decomposition.
    shape : tuple
        Hankel matrix dimensions, i.e. shape of the underlying numpy array.

    """
    def __init__(self, data, check_pd):
        self._data = data
        self._det = None
        self._chol = None
        self.shape = self._data.shape
        if check_pd:
            self.chol()    # fails if matrix is not p.d.

    @classmethod
    def from_moments(cls, mom, kind='lower', check_pd=False):
        """
        Initialize Hankel matrix from a set of moments. The entries are determined by the number of moments and the `kind`-parameter.

        Parameters
        ----------
        mom : array
            Set of moments. If `len(mom)` is even, the resulting Hankel-matrix depends on the parameter `kind`.
        kind : str, optional
            Kind of Hankel moment matrix, must be either 'lower' (default) or 'upper'. If the number of given moments, say n, is even and kind=='lower', the moments m[0]...m[n-2] are used to assemble the Hankel matrix. If kind=='upper', the Hankel matrix conatains the moments m[1]...m[n-1]. In cases where the number of moments is odd, the choice of `kind` has no effect.
        check_pd : bool, optional
            Indicates if positive definiteness of the Hankel matrix is to be checked. The default value is `False`.

        """
        valid_kinds = ['lower', 'upper']
        if kind not in valid_kinds:
            raise ValueError("Got invalid parameter kind={0!s}; must be '{1:s}' or '{2:s}'.".format(kind, *valid_kinds))
        return HankelMatrix(hankel_matrix(mom, kind), check_pd)

    def det(self):
        """
        Determinant of the Hankel moment matrix. If not yet assigned, a Cholesky decomposition is used for computation.

        Returns
        -------
        det : float
            Hankel determinant.

        """
        if self._det is None:
            self._det = np.prod(np.diag(self.chol()))**2
        return self._det

    def chol(self):
        """
        The upper right triangular matrix resulting from the Cholesky decomposition of the stored Hankel matrix.

        Returns
        -------
        chol : array
            2D-array representing the upper right triangular matrix resulting from a Cholesky-decomposition.

        """
        if self._chol is None:
            try:
                self._chol = la.cholesky(self._data, check_finite=False)
            except np.linalg.LinAlgError as e:
                raise ValueError("The given matrix is not positive definite: \n (LinAlgError) {0!s}.".format(e))
        return self._chol

    def matrix(self):
        """
        The stored Hankel matrix.

        Returns
        -------
        matrix : array
            2D-array representing the Hankel moment matrix.

        """
        return self._data

    def update(self, mom, kind, check_pd):
        """
        Update Hankel matrix with given moments. The given moments are presumed to be consistent with `self.shape`.

        Parameters
        ----------
        mom : array
            Set of moments. If `len(mom)` is even, the resulting Hankel-matrix depends on the parameter `kind`.
        kind : str, optional
            Kind of Hankel moment matrix, must be either 'lower' (default) or 'upper', see `HankelMatrix.from_moments`.
        check_pd : bool, optional
            Indicates if positive definiteness of the Hankel matrix is to be checked. The default value is `False`.

        """
        valid_kinds = ['lower', 'upper']
        if kind not in valid_kinds:
            raise ValueError("Got invalid parameter kind={0!s}; must be '{1:s}' or '{2:s}'.".format(kind, *valid_kinds))
        hankel_matrix(mom, kind=kind, inplace=True, A=self._data)
        self._det = None
        self._chol = None
        if check_pd:
            self.chol()

    def copy(self):
        """
        Return copy of this object.

        """
        c  = HankelMatrix(self._data.copy(), check_pd=False)
        if self._chol is not None:
            c._chol = self._chol.copy()
        else:
            c._chol = None
        c._det = self._det
        return c



    def __call__(self):
        """
        The stored Hankel matrix, see `HankelMatrix.matrix`-method.

        """
        return self.matrix()

    def __add__(self, other):
        """
        Sum of two Hankel matrices. Positive definiteness of the operands is presumed.

        Parameter

        """
        # If the input matrices are p.d., so is the sum.
        return HankelMatrix(self._data + other._data, check_pd=False)

    def __sub__(self, other):
        """
        Difference of two Hankel matrices.

        """
        # Check positive definiteness of difference
        return HankelMatrix(self._data - other._data, check_pd=True)

    def __eq__(self, other):
        """
        Check equality of two Hankel matrices.

        """
        return np.all(self._data == other._data)

    def __getitem__(self, idx):
        """
        Get elements or slices of Hankel matrix.

        """
        return self._data[idx]

