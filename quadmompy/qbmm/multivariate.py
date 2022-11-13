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

import abc
import warnings
from itertools import product as itprod
import numpy as np
from quadmompy.core.quadrature import ConditionalQuadrature
from quadmompy.qbmm.qbmm import Qbmm
from quadmompy.qbmm.univariate import UnivariateQbmm


class MultivariateQbmm(Qbmm):
    """
    Base class of all multivariate quadrature-based moment methods (QBMMs). It also serves as an interface to dynamically create instances of subclasses.

    Parameters
    ----------
    n_dims : int
        Number of dimensions.

    Attributes
    ----------
    n_dims : int
        Number of dimensions.

    """
    def __init__(self, n_dims, **kwargs):
        super().__init__(**kwargs)
        self.n_dims = n_dims

    @classmethod
    def new(cls, qbmm_type, qbmm_setup, **kwargs):
        """
        Create new instance of a specified `MultivariateQbmm`-subtype by calling the `new`-method implemented in the subclass.

        Parameters
        ----------
        qbmm_type : str or type
            `MultivariateQbmm`-subtype, either directly as type or as string or by `name`-variable implemented in subclass.
        qbmm_setup : dict
            Dictionary containing the parameters required to initialize `qbmm_type`-object.

        Returns
        -------
        new : MultivariateQbmm
            Instance of the specified subclass through the subclass's `new`-method.

        """
        return super().new(qbmm_type, qbmm_setup, **kwargs)


class ConditionalQmom(MultivariateQbmm):
    """
    The conditional quadrature method of moments [:cite:label:`Yuan_2011`] for multivariate problems.

    Parameters
    ----------
    config1d : list
        List of `n_dims` setup dictionaries to initialize `UnivariateQbmm` object for each dimension.

    Attributes
    ----------
    qbmm1d : list
        List with length `n_dims` containing univariate moment inversion algorithms, i.e. objects of a subtype of `UnivariateQbmm`, for each dimension. The selected methods must be consistent with each other and with the CQMOM.

    References
    ----------
        +-------------+-------------------+
        | [Yuan_2011] | :cite:`Yuan_2011` |
        +-------------+-------------------+

    """
    name = 'CQMOM'
    def __init__(self, config1d, mom_corr=None, **kwargs):
        super().__init__(**kwargs)
        self.qbmm1d = [UnivariateQbmm.new(**conf) for conf in config1d]
        self.mom_corr = mom_corr

    @classmethod
    def new(cls, qbmm_setup, **kwargs):
        """
        Return instance of `ConditionalQmom` class (necessary for the dynamic selection from parent class to work).

        Parameters
        ----------
        setup : dict
            Setup dictionary containing parameters for basic moment inversion algorithm.

        Returns
        -------
        new : ConditionalQmom
            New `ConditionalQmom` object.

        """
        return cls(**qbmm_setup)

    def moment_inversion(self, mom):
        """
        Compute multivariate quadrature using the CQMOM [:cite:label:`Yuan_2011`], given an `n_dims`-dimensional moment set.

        Parameters
        ----------
        mom : array
            A moment set with `n_dims` dimensions. It is left to the user to ensure that all necessary moments are included in the given set of moments.

        Returns
        -------
        quad : ConditionalQuadrature
            The computed multidimensional quadrature wrapped in a `ConditionalQuadrature` object.

        Raises
        ------
        ValueError
            If the dimensions of the given set of moments do not match `n_dims`.

        References
        ----------
            +-------------+-------------------+
            | [Yuan_2011] | :cite:`Yuan_2011` |
            +-------------+-------------------+

        """
        # Check consistency of given moment set and `n_dims`
        if len(mom.shape) != self.n_dims:
            msg = "Dimensionality of the given moment set ({0:d}) and n_dims ({1:d}) are not consistent".format(len(mom.shape), self.n_dims)
            raise ValueError(msg)

        # Initialize `ConditionalQuadrature`
        n_nodes = [self.qbmm1d[i].nodes_max(mom.shape[i]) for i in range(self.n_dims)]
        quad = ConditionalQuadrature.empty(n_nodes)

        # TODO: All the repeated initializations and copying should probably be avoided for the sake of efficiency
        # Lists for weight and Vandermonde matrices
        dim = 0
        V = []
        R = []

        # Compute conditional moments and conditional quadrature for each dimension given the quadrature
        # of a 'previous' dimension. This procedure is a generalization of Eqs. (27)-(32) in Ref. [Yuan_2011]
        while dim < self.n_dims:
            # Initialize matrix of conditional moments
            mom_order_max = tuple([nn for nn in n_nodes[:dim]] + [mom.shape[dim] - 1])
            mom_order = tuple(map(slice ,mom_order_max[:dim]))+tuple([slice(1, None)] + [0]*(self.n_dims - dim - 1))
            cmom = np.zeros(mom_order_max)
            cmom[mom_order[:dim]] = mom[mom_order]
            try:
                # This should work with standard-QMOM in each dimension
                cmom[mom_order[:dim]] = mom[mom_order]
            except ValueError:
                # TODO: make this work with other QBMMs, particularly EQMOM this may fail
                cmom[tuple(map(slice, mom.shape))] = mom[mom_order]
                idx_0 = np.argwhere(cmom == 0.)
                for i in idx_0:
                    cmom[i] = quad.reconstruct_moment(i)

            d = 0
            while d < dim:
                # Dimension of Vandermonde-matrix
                vdim = tuple(n_nodes[:d]) + (n_nodes[d],)*2
                if len(V) <= d:
                    # Initialize and compute weight and Vandermonde matrix
                    V.append(np.zeros(vdim))
                    R.append(np.zeros(vdim))
                    node = list(itprod(*tuple(tuple([i for i in range(nn)]) for nn in n_nodes[:d])))
                    for n in node:
                        idx = (d,) + n
                        try:
                            V[-1][n] = np.vander(quad.xi_cond[idx], n_nodes[d], increasing=True).T
                            R[-1][n] = np.diag(quad.w_cond[idx])
                        except ValueError as err:
                            if V[-1][n].shape[0] == 1:
                                V[-1][n] = np.ones((1,1))
                                R[-1][n] = np.ones((1,1))
                            else:
                                raise err

                # Solve Vandermonde System
                cmom = np.moveaxis(cmom, [i for i in range(1, len(cmom.shape) - 1)], \
                        [i for i in range(len(cmom.shape) - 2)])
                A = np.matmul(V[d], R[d])
                try:
                    # `cmom` corresponds to Y in Eq. (30) (Ref. [Yuan_2011]]) if `d < dim - 1`
                    # and to the actual conditional moments in the last step, i.e. `d = dim - 1`
                    cmom = np.linalg.solve(A, cmom)
                except np.linalg.LinAlgError:
                ### TODO: This is done in cases where at least one weight is zero; check if this is safe
                    warnings.warn("NumPy's linalg.solve-function failed to solve the Vandermonde system, using linalg.lstsq instead.")
                    if len(cmom.shape) > 2:
                        dims_m2 = np.ndindex(cmom.shape[:-2])
                        start = len(cmom.shape) - len(A.shape)
                        for d_ in dims_m2:
                            cmom[d_], r, _, _ = np.linalg.lstsq(A[d_[start:]], cmom[d_], rcond=None)
                    else:
                        cmom, r, _, _ = np.linalg.lstsq(A, cmom, rcond=None)
                d += 1

            # With the conditional moments, compute quadrare for each previously computed node
            node = [[dim]] + [[i for i in range(nn)] for nn in n_nodes[:dim]]
            node = list(itprod(*node))
            for n in node:
                mom_idx = n[1:] + (slice(None),)
                quad.xi_cond[n], quad.w_cond[n] = self.qbmm1d[dim].moment_inversion(np.append(1., cmom[mom_idx]))
                ok = not np.all(np.isfinite(quad.xi_cond[n]))
                if not ok and self.mom_corr:
                    cmom[mom_idx] = self.mom_corr(cmom[mom_idx])
                    pass

            dim += 1

        # Update ConditionalQuadrature (compute general ND-quadrature from conditional quadrature) and return
        quad.update()
        return quad
