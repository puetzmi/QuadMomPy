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
Quadrature (both univariate and multivariate) data structures required by several QBMMs.

"""
import numpy as np
import abc


class Quadrature(metaclass=abc.ABCMeta):
    """
    The abstract Quadrature base class.

    Parameters
    ----------
    xi : array
        Quadrature nodes.
    w : array
        Quadrature weights.

    Attributes
    ----------
    xi : array
        Quadrature nodes.
    w : array
        Quadrature weights.
    n_nodes : int
        Number of quadrature nodes.
    n_dims : int
        Number of dimensions.

    """
    def __init__(self, xi, w):
        self.xi = xi
        try:
            n_nodes, n_dims = xi.shape
        except ValueError: # one-dimensional quadrature
            n_nodes = xi.shape[0]
            n_dims = 1
        self.w = w
        if self.w.shape[0] != n_nodes:
            raise ValueError("Length of vector of weights must equal the number of dimensions")
        self.n_nodes = n_nodes
        self.n_dims = n_dims

    @classmethod
    @abc.abstractmethod
    def empty(cls, *args):
        pass

    def __getitem__(self, node_idx):
        nodes = self.xi[node_idx]
        weight = self.w[node_idx]
        return nodes, weight

    @abc.abstractmethod
    def reconstruct(self, order, factor=1):
        """
        Reconstruct single moment with given order from quadrature.

        Parameters
        ----------
        order : int
            The moment order.
        factor : array or float, optional
            Additional factor such as a source term. If it is an array, it must have a shape consistent with `xi`.

        Returns
        -------
        moments : array
            Reconstructed moments of order `order`.

        """
        pass

    def reconstruct_all(self, max_order, factor=1):
        """
        Reconstruct all (possibly multivariate) moments up to given order.

        Parameters
        ----------
        max_order : int or sequence of ints
            The maximum moment order in each dimension.
        factor : array or float, optional
            Additional factor such as a source term. If it is an array, it must have a shape consistent with xi.

        Returns
        -------
        moments : array
            Reconstructed moments of shape `max_order`.

        """
        moments = np.empty(tuple(max_order))
        for order in np.ndindex(max_order):
            moments[order] = self.reconstruct(order, factor=factor)
        return moments


class OneDQuadrature(Quadrature):
    """
    Univariate/one-dimensional quadrature.

    Parameters
    ----------
    xi : array
        1D-array with quadrature nodes.
    w : array
        1D-array with quadrature weights.

    """
    def __init__(self, xi, w):
        super().__init__(xi, w)

    @classmethod
    def empty(cls, n_nodes):
        """
        Create a OneDQuadrature instance. Arrays for nodes and weights are allocated but not initialized.

        Parameters
        ----------
        n_nodes : int
            Number of quadrature nodes.

        Returns
        -------
        quadrature : OneDQuadrature
            A OneDQuadrature object with n_nodes nodes.

        """
        xi = np.empty(n_nodes)
        w = np.empty(n_nodes)
        return cls(xi, w)

    def reconstruct(self, moment_order, factor=1):
        return (factor*self.xi**order)@self.w

    def __iter__(self):
        return iter((self.xi, self.w))


class NDQuadrature(Quadrature):
    """
    N-dimensional quadrature.

    Parameters
    ----------
    xi : array
        Quadrature nodes of shape `(n_nodes, n_dims)` representing `n_nodes` `n_dims`-dimensional vectors.
    w : array
        Quadrature weights of shape `(n_nodes,)`.

    """
    def __init__(self, xi, w):
        super().__init__(xi, w)

    @classmethod
    def empty(cls, n_nodes, n_dims):
        """
        Create an `NDQuadrature` instance. Arrays for nodes and weights are allocated but not initialized.

        Parameters
        ----------
        n_nodes : int
            Total number of quadrature nodes.
        n_dims : int
            Number of dimensions.

        Returns
        -------
        quadrature : NDQuadrature
            An `NDQuadrature` object with `n_nodes` nodes and `n_dims` dimensions.

        """
        xi = np.empty((n_nodes, n_dims))
        w = np.empty(n_nodes)
        return cls(xi, w)

    def reconstruct(self, moment_order, factor=1):
        k = np.array(moment_order)
        return (factor*np.prod(self.xi**k, axis=-1))@self.w


class _CondBaseStruct:
    """
    Data structure to store the conditional quadrature. Contains a list of arrays with shape (N1,N2,..,Nd) in the dth dimension.

    Parameters
    ----------
    nodes_per_dim : sequence of ints
        Number of nodes in each dimension.

    """
    def __init__(self, n_nodes_per_dim):
        self.data = []
        n_dims = len(n_nodes_per_dim)
        for i in range(n_dims):
            self.data.append(np.empty(tuple(n_nodes_per_dim[:i+1])))

    def __getitem__(self, idx):
        """
        Get element(s) at given index. Slicing is supported.

        Parameters
        ----------
        idx : int or sequence of ints.
            Index of the element(s). The first int is the dimension, the following indices refer to nodes.

        Return
        ------
        elements : array or float
            The element(s) at the given index.

        Examples
        --------
        Structure with 3, 2 and 2 nodes in the first, second and third dimension, respectively:
        >>> nodes = _CondBaseStruct((3, 2, 2))

        Set nodes in the first dimension:
        >>> nodes[0] = np.array([0.1, 0.5, 0.6])

        Set 3x2 nodes in the second dimension:
        >>> nodes[1] = np.array([[1, 2], [3, 4], [5, 6]])

        Get both nodes in the second dimension (1) at the third node (3) in the first dimension:
        >>> nodes[1,2]
        [3. 3.]

        ... or only the second node (1) in the second dimension (1) at first node (0) in the first dimension:
        >>> nodes[1,0,1]
        2.

        """
        try:
            if len(idx) > 1:
                return self.data[idx[0]][idx[1:]]
            return self.data[idx[0]]
        except TypeError:
            return self.data[idx]

    def __setitem__(self, idx, val):
        """
        Set element(s). Slicing is supported.

        Parameters
        ----------
        idx : int or sequence of ints.
            Index of the element(s). The first int is the dimension, the following indices refer to nodes.
        val : array or float
            New value(s) of the element(s) at the given index. The shape must be consistent with the shape in the given dimension.

        Examples
        --------
        Structure with 3, 2 and 2 nodes in the first, second and third dimension, respectively:
        >>> nodes = _CondBaseStruct((3, 2, 2))

        Set nodes in the first dimension:
        >>> nodes[0] = np.array([0.1, 0.5, 0.6])

        Set 3x2 nodes in the second dimension:
        >>> nodes[1] = np.array([[1, 2], [3, 4], [5, 6]])

        Set only one element:
        >>> nodes[1,2,0] = 0.
        >>> nodes[1]
        [[1. 2.]
         [3. 4.]
         [0. 6.]]

        """
        try:
            if len(idx) > 1:
                self.data[idx[0]][idx[1:]] = val
                return
            self.data[idx[0]] = val
        except TypeError:
            self.data[idx] = val


class ConditionalQuadrature(NDQuadrature):
    """
    N-dimensional quadrature required for the conditional quadrature method of moments (CQMOM) [:cite:label:`Yuan_2011`].

    Parameters
    ----------
    xi : array
        Quadrature nodes of shape `(n_nodes, n_dims)`, where `n_nodes` is the product of all elements in `n_nodes_per_dim`.
    w : array
        Quadrature weights of shape `(n_nodes,)`, where `n_nodes` is the product of all elements in `n_nodes_per_dim`.
    n_nodes_per_dim : sequence of ints
        Number of nodes in each dimension.
    init_cond : bool, optional
        If True, the conditional quadrature is updated during instantiation based on `xi` and `w`. Default is `True`.

    Attributes
    ----------
    n_nodes_per_dim : sequence of ints
        Number of nodes in each dimension.
    xi_cond : _CondBaseStruct
        Conditional quadrature nodes wrapped in `_CondBaseStruct`-object.
    w_cond : _CondBaseStruct
        Conditional quadrature weights wrapped in `_CondBaseStruct`-object.

    References
    ----------
        +-------------+-------------------+
        | [Yuan_2011] | :cite:`Yuan_2011` |
        +-------------+-------------------+

    """
    def __init__(self, xi, w, n_nodes_per_dim, init_cond=True):
        super().__init__(xi, w)
        self.n_nodes_per_dim = n_nodes_per_dim
        self.xi_cond = _CondBaseStruct(n_nodes_per_dim)
        self.w_cond = _CondBaseStruct(n_nodes_per_dim)

    @classmethod
    def empty(cls, n_nodes_per_dim):
        """
        Create an instance of `ConditionalQuadrature`. Arrays for nodes and weights are allocated but not properly initialized. So are data structures for the conditional quadrature.

        Parameters
        ----------
        n_nodes_per_dim : sequence of ints
            Number of quadrature nodes in each dimension.

        Returns
        -------
        quadrature : ConditionalQuadrature
            A `ConditionalQuadrature` object with `n_nodes` nodes and `n_dims` dimensions.

        """
        n_nodes = np.prod(n_nodes_per_dim)
        n_dims = len(n_nodes_per_dim)
        xi = np.empty((n_nodes, n_dims))
        w = np.empty(n_nodes)
        return cls(xi, w, n_nodes_per_dim, init_cond=False)

    def update(self):
        """
        Update the general multivariate quadrature from the conditional quadrature. This method must be called explicitly after updating the conditional quadrature, e.g. after a moment inversion.

        """
        self.w[:,] = 1.
        for i in range(self.n_dims):
            rep = np.prod(self.n_nodes_per_dim[i+1:])
            self.xi[:,i] = self.xi_cond[i].repeat(rep)
            self.w *= self.w_cond[i].repeat(rep)
