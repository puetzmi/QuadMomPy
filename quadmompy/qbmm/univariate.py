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
from math import sqrt, factorial
import numpy as np
from scipy.special import roots_hermite, roots_genlaguerre, roots_jacobi
from scipy.linalg import toeplitz
from scipy.special import factorial as spfactorial
from quadmompy.core.inversion import MomentInversion
from quadmompy.core.quadrature import OneDQuadrature
from quadmompy.qbmm.qbmm import Qbmm
from quadmompy.qbmm import _eqmom_root_search as eqroots
from quadmompy.moments.special import laplace_moments
from scipy.special import gamma as gamma_func
from scipy.special import beta as beta_func
from scipy import optimize


class UnivariateQbmm(Qbmm):
    """
    Base class of all univariate quadrature-based moment methods (QBMMs). It also serves as an interface to dynamically create instances of subclasses.

    Parameters
    ----------
    inv_type : type or str
        Basic moment inversion algorithm, can be either subtype of `MomentInversion` or an associated string, see class `MomentInversion`
    setup : dict
        Dictionary with parameters needed to initialize the basic `MomentInversion`-subclass.

    Attributes
    ----------
    inversion : MomentInversion
        Basic moment inversion algorithm to compute quadrature from moments.

    """
    def __init__(self, inv_type, inv_setup={}, **kwargs):
        try:
            self.inversion = inv_type(**inv_setup)
        except TypeError:
            self.inversion = MomentInversion.new(inv_type, **inv_setup)

    @abc.abstractmethod
    def nodes_max(cls, nmom):
        """
        The maximum number of nodes given `nmom` moments.

        Parameters
        ----------
        nmom : int
            Number of moments for quadrature computation.

        Returns
        -------
        nodes_max : int
            Maximum number of quadrature nodes.

        """
        pass


class Qmom(UnivariateQbmm):
    """
    Standard quadrature method of moments (QMOM).

    This class is the implementation of the QMOM [:cite:label:`McGraw_1997`]. It computes N quadrature nodes and weights from a set of 2N moments.

    Attributes
    ----------
    inversion : MomentInversion
        Basic moment inversion algorithm to compute quadrature from moments, see class `MomentInversion`.

    References
    ----------
        +---------------+---------------------+
        | [McGraw_1997] | :cite:`McGraw_1997` |
        +---------------+---------------------+

    """
    name = 'QMOM'
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def new(cls, qbmm_setup, **kwargs):
        """
        Return instance of `Qmom` class (necessary for the dynamic selection from parent class to work).

        Parameters
        ----------
        qbmm_setup : dict
            Setup dictionary containing parameters for basic moment inversion algorithm.

        Returns
        -------
        new : Qmom
            New `Qmom` object.

        """
        return cls(**qbmm_setup)

    def moment_inversion(self, mom, radau_node=None):
        if radau_node is not None:
            self.inversion.radau_node = radau_node
        return OneDQuadrature(*self.inversion(mom))

    def nodes_max(self, nmom):
        return nmom//2


class GaGQmom(UnivariateQbmm):
    """
    Gauss/anti-Gauss quadrature method of moments (GaG-QMOM).

    This class is the implementation of the GaG-QMOM [:cite:label:`Puetz_2022`] based on the anti-Gaussian quadrature due to Laurie [:cite:label:`Laurie_1996`]. It computes 2N-1 quadrature nodes and weights from a set of 2N moments.

    Attributes
    ----------
    inversion : MomentInversion
        Basic moment inversion algorithm to compute quadrature from moments, see class `MomentInversion`.

    References
    ----------
        +---------------+---------------------+
        | [Puetz_2022]  | :cite:`Puetz_2022`  |
        +---------------+---------------------+
        | [Laurie_1996] | :cite:`Laurie_1996` |
        +---------------+---------------------+

    """
    name = 'GaGQMOM'
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Check if Gauss-Radau or Gauss-Lobatto quadrature is
        # required and assign inversion function accordingly
        if self.inversion.radau:
            self.inv_func = self._inv_radau
        elif self.inversion.lobatto:
            self.inv_func = self._inv_lobatto
        else:
            self.inv_func = self._inv_gauss

    @classmethod
    def new(cls, qbmm_setup, **kwargs):
        """
        Return instance of `GaGQmom` class (necessary for the dynamic selection from parent class to work).

        Parameters
        ----------
        setup : dict
            Setup dictionary containing parameters for basic moment inversion algorithm.

        Returns
        -------
        new : GaGQmom
            New `GaGQmom` object.

        """
        return cls(**qbmm_setup)

    def _inv_gauss(self, mom):
        """
        Computation of the averaged (2N-1)-node standard Gauss/anti-Gauss quadrature given 2N moments.

        Parameters
        ----------
        mom : array
            Univariate set of 2N moments.

        Returns
        -------
        quadrature : OneDQuadrature
            2N-1 quadrature nodes and weights wrapped in `OneDQuadrature` object.

        """
        nmom = len(mom)
        # Initialize arrays for 2N-1 quadrature nodes and weights
        xi = np.empty(nmom - 1)
        w = xi.copy()

        # Compute recurrence coefficients
        alpha, beta = self.inversion.recurrence_coeffs(mom)

        # For the N-node GaG-quadrature the last beta-coefficient must be doubled
        beta[-1] *= 2

        # Compute (N-1)-node Gaussian quadrature
        xi[1::2], w[1::2] = self.inversion.quad_from_rc(alpha[:-1], beta[:-1])

        # Compute N-node anti-Gaussian quadrature
        xi[::2], w[::2] = self.inversion.quad_from_rc(alpha, beta)

        # The final weights correspond to the average of the Gaussian and anti-Gaussian quadrature
        w *= 0.5

        return OneDQuadrature(xi, w)

    # TODO Add Radau quadrature for Stieltjes problems
    def _inv_radau(self, mom):
        """
        Computation of the averaged (2N-1)-node Gauss/anti-Gauss quadrature given 2N moments.

        Parameters
        ----------
        mom : array
            Univariate set of 2N moments.

        Returns
        -------
        quadrature : OneDQuadrature
            2N-1 quadrature nodes and weights wrapped in `OneDQuadrature` object.

        """
        raise NotImplementedError()

    def _inv_lobatto(self, mom):
        """
        Computation of the averaged (2N-1)-node Gauss/anti-Gauss-Lobatto quadrature given 2N moments.

        Parameters
        ----------
        mom : array
            Univariate set of 2N moments.

        Returns
        -------
        quadrature : OneDQuadrature
            2N-1 quadrature nodes and weights wrapped in `OneDQuadrature` object.

        """
        nmom = len(mom)
        # Initialize arrays for 2N-1 quadrature nodes and weights
        xi = np.empty(nmom - 1)
        w = xi.copy()

        # Compute recurrence coefficients
        alpha, beta = self.inversion.recurrence_coeffs(mom)

        # For the (N+1)-node GaG-quadrature the last beta-coefficient must be doubled
        beta[-1] *= 2

        # Compute N-node Gauss-Lobatto quadrature
        xi[::2], w[::2] = self.inversion.quad_from_rc(alpha[:-1], beta[:-1])

        # Compute (N+1)-node anti-Gauss-Lobatto quadrature
        xi_, w_ = self.inversion.quad_from_rc(alpha, beta)
        xi[1::2] = xi_[1:-1]
        w[1::2] = w_[1:-1]
        w[0] += w_[0]
        w[-1] += w_[-1]

        # The final weights correspond to the average of the Gaussian and anti-Gaussian quadrature
        w *= 0.5

        return OneDQuadrature(xi, w)

    def moment_inversion(self, mom):
        """
        Computation of the averaged selected type of Gauss/anti-Gauss quadrature given 2N moments.

        Parameters
        ----------
        mom : array
            Univariate set of 2N moments.

        Returns
        -------
        quadrature : OneDQuadrature
            2N-1 quadrature nodes and weights wrapped in `OneDQuadrature` object.

        """
        return self.inv_func(mom)

    def nodes_max(self, nmom):
        return nmom - 1


class ExtendedQmom(UnivariateQbmm):
    """
    The base class for the extended quadrature method of moments (EQMOM).

    This class contains the basic methods used by the extended quadrature method of moments (EQMOM) [:cite:label:`Yuan_2012`]. It also serves as an interface to create objects of specified subclasses characterized by the used kernel density function (KDF) type. Specific methods depending on the selected KDF must be implemented in the respective subclass.

    Parameters
    ----------
    n_ab : int
        Number of second quadrature nodes per first quadrature node.
    atol : float, optional
        Absolute tolerance used to find EQMOM-parameter sigma. Default is 1e-8.
    rtol : float, optional
        Relative tolerance used to find EQMOM-parameter sigma. Default is 1e-6.
    n_init : int
        Initial number of first quadrature nodes, needed for some initializations. Default is 5.

    Attributes
    ----------
    inversion : MomentInversion
        Basic moment inversion algorithm to compute quadrature from moments, see class `MomentInversion`.
    n_ab : int
        Number of second quadrature nodes per first quadrature node.
    atol : float
        Absolute tolerance used to find EQMOM-parameter sigma.
    rtol : float
        Relative tolerance used to find EQMOM-parameter sigma.
    n_init : int
        Initial number of first quadrature nodes, needed for some initializations.
    xi_first : array
        Abscissas of the first quadrature.
    w_first : array
        Weights of the first quadrature.
    xi_second : array
        Array with shape `(n_ab, len(xi_first))` containing abscissas of the second quadrature for each first quadrature node.
    w_second : array
        Array with shape `(n_ab, len(xi_first))` containing weights of the second quadrature for each first quadrature node.
    sigma : float
        KDF-specific EQMOM parameter.
    A_coeffs : array
        Constant factors of sigma in A-matrix (map m* -> m), for details see [:cite:label:`Yuan_2012`].
    Ainv_coeffs : array
        Constant factors of sigma in inverse of A (map m -> m*), for details see [:cite:label:`Yuan_2012`]).
    sigma_pow : array
        Sigma powers corresponding to elements in A and Ainv.

    References
    ----------
        +-------------+-------------------+
        | [Yuan_2012] | :cite:`Yuan_2012` |
        +-------------+-------------------+

    """
    name = 'EQMOM'
    def __init__(self, n_ab, atol=1e-8, rtol=1e-6, n_init=5, **kwargs):
        super().__init__(**kwargs)
        self.n_ab = n_ab
        self.atol = atol
        self.rtol = rtol
        self.n_init = n_init
        self.xi_first = None
        self.w_first = None
        self.xi_second = None
        self.w_second = None
        self.sigma = 0.
        nmom = 2*self.n_init + 1
        self.A_coeffs = self._init_A_coeffs(nmom)
        self.Ainv_coeffs = self._init_Ainv_coeffs(nmom)
        self.sigma_pow = self._init_sigma_pow(nmom)

    @classmethod
    def new(cls, qbmm_setup, **kwargs):
        """
        Create a new object of a subclass based on the specified KDF-type.

        Parameters
        ----------
        qbmm_setup : dict
            Setup dictionary with the KDF-type and additional parameters passed to `UnivariateQbmm` and the selected subclass.

        Returns
        -------
        new : ExtendedQmom
            Instance of the specified subclass.

        """
        eqmom_type = qbmm_setup["type"]
        try:
            if issubclass(eqmom_type, cls):
                return eqmom_type.new(**kwargs)
        except TypeError:
            if eqmom_type[-5:] != "EQMOM":
                eqmom_type += "EQMOM"
            return super().new(eqmom_type, qbmm_setup, **kwargs)

    def nodes_max(self, nmom):
        return self.n_ab*((nmom - 1)//2)

    def moment_inversion(self, mom, sort_combined=False):
        """
        Basic moment inversion procedure of the EQMOM.

        Parameters
        ----------
        mom : array
            Set of 2N + 1 moments where N is the number of primary nodes and weights.
        sort_combined : bool, optional
            If true, the combined nodes are sorted by the abscissae in ascending order, default is `False`.

        Returns
        -------
        quadrature : OneDQuadrature
            Resulting nodes and weights wrapped in `OneDQuadrature` object.

        """
        # First check if the current sigma is already within the specified tolerance
        try:
            found_sigma = abs(self._target_function(self.sigma, mom)) < self.atol
        except np.linalg.LinAlgError:
            found_sigma = False

        # Then try to compute sigma analytically
        if not found_sigma:
            s = self._sigma_analyt(mom)
            if s is not None:
                self.sigma = s
                ms = self.m2ms(mom[:-1], self.sigma)
                self.xi_first, self.w_first = self.inversion(ms)

        # If that does not work find root using the algorithm implemented in `find_sigma`
            else:
                self.sigma, xi, w = self.find_sigma(mom)
                n = len(xi)
                self.xi_first[:n] = xi
                self.w_first[:n] = w
                self.xi_first[n:] = self.w_first[n:] = 0.

        # Return first quadrature if sigma is (approximately) zero; set second quadrature accordingly
        if abs(self.sigma) < self.atol:
            self.xi_second = np.zeros((self.n_ab, len(self.xi_first)))
            self.xi_second[0,:] = self.xi_first
            self.w_second = np.zeros((self.n_ab, len(self.w_first)))
            self.w_second[0,:] = 1.
            return self.xi_first, self.w_first

        # Otherwise compute second quadrature
        self.xi_second, self.w_second = self.second_quad(self.xi_first, self.sigma, self.n_ab)
        return self._combined_quad(self.w_first, self.xi_second, self.w_second, sort_combined)

    def find_sigma(self, mom):
        r"""
        Routine to find the EQMOM parameter :math:`\sigma`, which is the root of the target function :math:`\mathbf{J}(\sigma)`, see [:cite:label:`Yuan_2012`].

        Parameters
        ----------
        mom : array
            Realizable set of moments.

        Returns
        -------
        sigma : float
            The found root of the target function.
        xi : array
            Nodes of the first quadrature.
        w : array
            Weights of the first quadrature.

        References
        ----------
            +-------------+-------------------+
            | [Yuan_2012] | :cite:`Yuan_2012` |
            +-------------+-------------------+

        """
        bounds = self._sigma_bounds(mom)
        sigma = optimize.ridder(self._target_function, *bounds, args=(mom,))
        xi, w = self.inversion(self.m2ms(mom[:-1], sigma))
        return sigma, xi, w

    @abc.abstractmethod
    def _init_sigma_pow(self, nmom):
        """
        Initialize array containing the powers of :math:`\sigma` in the lower triangular matrices :math:`\mathbf{A}(\sigma)` and :math:`\mathbf{A}^{-1}(\sigma)`, see [:cite:label:`Yuan_2012`].

        Parameters
        ----------
        nmom : int
            Number of moments.

        Returns
        -------
        sigma_pow: array
            Powers of sigma with shape (nmom, nmom).

        References
        ----------
            +-------------+-------------------+
            | [Yuan_2012] | :cite:`Yuan_2012` |
            +-------------+-------------------+

        """
        pass

    @abc.abstractmethod
    def _init_A_coeffs(self, nmom):
        """
        Initialize matrix with the constant factors in the lower triangular matrix :math:`\mathbf{A}(\sigma)` (map :math:`\mathbf{m}^{*} \\rightarrow \mathbf{m}`), see [:cite:label:`Yuan_2012`].

        Parameters
        ----------
        size : int
            Size of square matrix A.

        Returns
        -------
        A_coeffs: array
            Constant coefficients in the matrix A with shape (size, size).

        References
        ----------_
            +-------------+-------------------+
            | [Yuan_2012] | :cite:`Yuan_2012` |
            +-------------+-------------------+

        """
        pass

    @abc.abstractmethod
    def _init_Ainv_coeffs(self, nmom):
        """
        Initialize matrix with the constant factors in the lower triangular matrix :math:`\mathbf{A}^{-1}(\sigma)` (map :math:`\mathbf{m} \\rightarrow \mathbf{m}^{*}`), see [:cite:label:`Yuan_2012`].

        Parameters
        ----------
        size : int
            Size of square matrix Ainv.

        Returns
        -------
        Ainv_coeffs: array
            Constant coefficients in the matrix Ainv with shape (size, size).

        References
        ----------
            +-------------+-------------------+
            | [Yuan_2012] | :cite:`Yuan_2012` |
            +-------------+-------------------+

        """
        pass

    def A(self, sigma, nmom):
        """
        Assemble Matrix :math:`\mathbf{A}(\sigma)`, the linear map :math:`\mathbf{m}^{*} \\rightarrow \mathbf{m}`, see [:cite:label:`Yuan_2012`].

        Parameters
        ----------
        sigma : float
            KDF parameter.
        nmom : int
            Number of moments

        Returns
        -------
        A : array
            2D-array representing the matrix `A`.

        References
        ----------
            +-------------+-------------------+
            | [Yuan_2012] | :cite:`Yuan_2012` |
            +-------------+-------------------+

        """
        return self.A_coeffs[:nmom,:nmom]*sigma**self.sigma_pow[:nmom,:nmom] # works for most KDF-types

    def Ainv(self, sigma, nmom):
        """
        Assemble Matrix :math:`\mathbf{A}^{-1}(\sigma)`, the linear map :math:`\mathbf{m} \\rightarrow \mathbf{m}^{*}`, see [:cite:label:`Yuan_2012`].

        Parameters
        ----------
        sigma : float
            KDF parameter.
        nmom : int
            Number of moments

        Returns
        -------
        Ainv : array
            2D-array representing the matrix `Ainv`.

        References
        ----------
            +-------------+-------------------+
            | [Yuan_2012] | :cite:`Yuan_2012` |
            +-------------+-------------------+

        """
        return self.Ainv_coeffs[:nmom,:nmom]*sigma**self.sigma_pow[:nmom,:nmom] # works for most KDF-types

    def _target_function(self, sigma, mom):
        """
        Target function :math:`\mathbf{J}(\sigma)` (see [:cite:label:`Yuan_2012`]) of which the smallest root is :math:`\sigma`.

        Parameters
        ----------
        sigma : float
            Current sigma.
        mom : array_like
            Set of 2N + 1 moments where N is the number of primary nodes and weights.

        Returns
        -------
        Jsigma : float
            Function value of current sigma.

        References
        ----------
            +-------------+-------------------+
            | [Yuan_2012] | :cite:`Yuan_2012` |
            +-------------+-------------------+

        """
        # The numbers next to the statements below correspond to the steps in section 3.5 in Ref. [Yuan_2012]
        _2n = len(mom) - 1
        ms = np.zeros(len(mom))
        ms[:-1] = self.m2ms(mom[:-1], sigma)                        # (1.)

        alpha, beta = self.inversion.recurrence_coeffs(ms[:-1])
        self.xi_first, self.w_first = \
                self.inversion.quad_from_rc(alpha, abs(beta))       # (2.)

        ms[-1] = self.w_first@(self.xi_first**_2n)                  # (3.)

        return self.ms2m(ms, sigma)[_2n] - mom[_2n]                 # (4.)

    def ms2m(self, ms, sigma):
        """
        Convert degenerated moments :math:`\mathbf{m}^{*}` to ordinary moments :math:`\mathbf{m}`, for details see [:cite:label:`Yuan_2012`].

        Parameters
        ----------
        ms : array
            Degenerated moments.
        sigma : float
            KDF parameter.

        Returns
        -------
        m : array
            Ordinary moments.

        References
        ----------
            +-------------+-------------------+
            | [Yuan_2012] | :cite:`Yuan_2012` |
            +-------------+-------------------+

        """
        nmom = len(ms)
        return self.A(sigma, nmom)@ms

    def m2ms(self, m, sigma):
        """
        Convert ordinary moments :math:`\mathbf{m}` to degenerated moments :math:`\mathbf{m}^{*}`, for details see [:cite:label:`Yuan_2012`].

        Parameters
        ----------
        m : array
            Ordinary moments.
        sigma : float
             KDF parameter.

        Returns
        -------
        ms : array
            Degenerated moments.

        References
        ----------
            +-------------+-------------------+
            | [Yuan_2012] | :cite:`Yuan_2012` |
            +-------------+-------------------+

        """
        nmom = len(m)
        return self.Ainv(sigma, nmom)@m

    @abc.abstractmethod
    def second_quad(self, xi_first, sigma, n_ab):
        """
        Compute second quadrature from the first quadrature and EQMOM-parameter :math:`\sigma`.

        Parameters
        ----------
        xi_first : array
            First quadrature nodes.
        sigma : float
            KDF parameter.
        n_ab : int
            Number of second quadrature nodes.

        Returns
        -------
        xi_second : array
            Array of shape (n_ab, len(xi_first)) containing quadrature nodes.
        w_second : array
            Array of shape (n_ab, len(xi_first)) containing quadrature weights.

        """
        pass

    @abc.abstractmethod
    def _sigma_bounds(self, mom):
        """
        KDF-specific bounds of the EQMOM-parameter :math:`\sigma`.

        Parameters
        ----------
        mom : array_like
            Set of moments.

        Returns
        -------
        sigma_bounds : tuple
            Bounds on sigma.

        """
        pass

    @abc.abstractmethod
    def _sigma_analyt(self, mom):
        """
        If possible, compute the parameter :math:`\sigma` analytically (depends on the number of quadrature nodes).

        Parameters
        ----------
        mom : array
            Set of moments.

        Returns
        -------
        sigma : float
            The EQMOM-parameter.

        """
        return None

    def _combined_quad(self, w_first, xi_second, w_second, sort=False):
        """
        Combine :math:`N_1`-node first quadrature and :math:`N_2`-node second quadrature into single :math:`N_1\\times N2`-node-quadrature.

        Parameters
        ----------
        w_first : array
            Weights of the first quadrature.
        xi_second : array
            Abscissae of the second quadrature.
        w_second : array
            Weights of the second quadrature.
        sort : bool, optional
            True if the nodes are supposed to be sorted by abscissae in ascending order (default: False).

        Returns
        -------
        xi : array
            Abscissae in flattened 1D-array with length `len(xi_first)*len(xi_second)`.
        w : array
            Combined weights in flattened 1D-array with length `len(xi_first)*len(xi_second)`.

        """
        w = (w_first*w_second).T.flatten()
        xi = xi_second.T.flatten()
        if not sort:
            return xi, w
        idx = np.argsort(xi)
        return xi[idx], w[idx]

    @abc.abstractmethod
    def ndf(self, xi):
        """
        Compute :math:`n(\\xi)` based on the selected KDF-type.

        Parameters
        ----------
        xi : array or float
            Abscissa(s).

        Returns
        -------
        ndf : array or float
            NDF-value(s) at xi.

        """
        pass


class GaussianEqmom(ExtendedQmom):
    """
    EQMOM with Gaussian KDFs.

    Parameters
    ----------
    kwargs :
        See base class `ExtendedQmom`.

    Attributes
    ----------
    inversion : MomentInversion
        Basic moment inversion algorithm to compute quadrature from moments, see class `MomentInversion`.
    n_ab : int
        Number of second quadrature nodes per first quadrature node.
    atol : float
        Absolute tolerance used to find EQMOM-parameter sigma.
    rtol : float
        Relative tolerance used to find EQMOM-parameter sigma.
    n_init : int
        Initial number of first quadrature nodes, needed for some initializations.
    xi_first : array
        Abscissas of the first quadrature.
    w_first : array
        Weights of the first quadrature.
    xi_second : array
        Array with shape `(n_ab, len(xi_first))` containing abscissas of the second quadrature for each first quadrature node.
    w_second : array
        Array with shape `(n_ab, len(xi_first))` containing weights of the second quadrature for each first quadrature node.
    sigma : float
        KDF-specific EQMOM parameter.
    A_coeffs : array
        Constant factors of sigma in A-matrix (map m* -> m), for details see [:cite:label:`Yuan_2012`]).
    Ainv_coeffs : array
        Constant factors of sigma in inverse of A (map m -> m*), for details see [:cite:label:`Yuan_2012`]).
    sigma_pow : array
        Sigma powers corresponding to elements in A and Ainv.
    xi_hermite : array
        Nodes of a Gauss-Hermite-quadrature to compute the second quadrature by simple linear transformation.
    w_hermite : array
        Weights of a Gauss-Hermite-quadrature to compute the second quadrature by simple linear transformation.

    References
    ----------
        +-------------+-------------------+
        | [Yuan_2012] | :cite:`Yuan_2012` |
        +-------------+-------------------+

    """
    name = 'gaussianEQMOM'
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        nmom = 2*self.n_init + 1
        self.xi_hermite, self.w_hermite_norm = roots_hermite(self.n_ab)
        self.w_hermite_norm /= sqrt(np.pi)

    @classmethod
    def new(cls, qbmm_setup, **kwargs):
        """
        Return instance of `GaussianEqmom` class (necessary for the dynamic selection from parent class to work).

        Parameters
        ----------
        qbmm_setup : dict
            Setup dictionary containing required parameters, see base classes `UnivariateQbmm` and `ExtendedQmom`.

        Returns
        -------
        new : GaussianEqmom
            New `GaussianEqmom` object.

        """
        return cls(**qbmm_setup)

    def find_sigma(self, mom):
        """
        Routine to find the EQMOM parameter :math:`\sigma`, which is the root of the target function :math:`\mathbf{J}(\sigma)`, see Ref. [:cite:label:`Yuan_2012`]. For the EQMOM with gaussian KDFs (Hamburger problem) the improved method based on moment realizability proposed in Ref. [:cite:label:`Pigou_2018`] is used.

        Parameters
        ----------
        mom : array
            Realizable set of moments.

        Returns
        -------
        sigma : float
            The found root of the target function.
        xi : array
            Nodes of the first quadrature.
        w : array
            Weights of the first quadrature.

        References
        ----------
            +--------------+--------------------+
            | [Yuan_2012]  | :cite:`Yuan_2012`  |
            +--------------+--------------------+
            | [Pigou_2018] | :cite:`Pigou_2018` |
            +--------------+--------------------+

        """
        bounds = self._sigma_bounds(mom)
        return eqroots.pigou_hamburger(bounds[1], bounds, mom, self.m2ms, self.ms2m, \
            self.inversion, self.atol, self.rtol)

    def _init_sigma_pow(self, nmom):
        """
        Initialize array containing the powers of :math:`\sigma` in the lower triangular matrices :math:`\mathbf{A}(\sigma)` and :math:`\mathbf{A}^{-1}(\sigma)` as described in Ref. [:cite:label:`Pigou_2018`] (Appendix B.1.2).

        Parameters
        ----------
        size : int
            Number of moments.

        Returns
        -------
        sigma_pow : array
            Powers of sigma with shape (nmom, nmom).

        References
        ----------
            +--------------+--------------------+
            | [Pigou_2018] | :cite:`Pigou_2018` |
            +--------------+--------------------+

        """
        # Compare Eqs. (B.4)-(B.7) in Ref. [Pigou_2018]
        sigma_pow = toeplitz([i*(1 - i%2) for i in range(nmom)], [0]*(nmom))
        return sigma_pow

    def _init_A_coeffs(self, size):
        """
        Initialize matrix with the constant factors in the lower triangular matrix :math:`\mathbf{A}(\sigma)` (map :math:`\mathbf{m}^{*} \\rightarrow \mathbf{m}`) as described in Ref. [:cite:label:`Pigou_2018`] (Appendix B.1.2).

        Parameters
        ----------
        size : int
            Size of square matrix A.

        Returns
        -------
        A_coeffs: array
            Constant factors in the matrix A with shape (size,size).

        References
        ----------
            +--------------+--------------------+
            | [Pigou_2018] | :cite:`Pigou_2018` |
            +--------------+--------------------+

        """
        A = np.identity(size)
        for i in range(size):
            for j in range(i%2,i,2):
                # Eq. (B.4) in Ref. [Pigou_2018]
                A[i,j] = factorial(i)/factorial((i-j)//2)/factorial(j)/2**((i-j)//2)
        return A

    def _init_Ainv_coeffs(self, size):
        """
        Initialize matrix with the constant factors in the lower triangular matrix :math:`\mathbf{A}^{-1}(\sigma)` (map :math:`\mathbf{m} \\rightarrow \mathbf{m}^{*}`) as described in Ref. [:cite:label:`Pigou_2018`] (Appendix B.1.2).

        Parameters
        ----------
        size : int
            Size of square matrix Ainv.

        Returns
        -------
        Ainv_coeffs: array
            Constant factors in the matrix Ainv with shape (size,size).

        References
        ----------
            +--------------+--------------------+
            | [Pigou_2018] | :cite:`Pigou_2018` |
            +--------------+--------------------+

        """
        Ainv = np.identity(size)
        for i in range(size):
            for j in range(i%2,i,2):
                # Eq. (B.5) in Ref. [Pigou_2018]
                Ainv[i,j] = factorial(i)/factorial((i-j)//2)/factorial(j)/(-2)**((i-j)//2)
        return Ainv

    def second_quad(self, xi_first, sigma, n_ab):
        """
        Compute second quadrature from the first quadrature and EQMOM-parameter :math:`\sigma`, which is, in the case of Gaussian KDFs, the parameter :math:`\sigma` of a normal distribution, i.e. the standard deviation. In this case, the second quadrature can be calculated by a simple linear transformation of the precomputed nodes and weights of a Gauss-Hermite quadrature.

        Parameters
        ----------
        xi_first : array or float
            First quadrature nodes.
        sigma : float
            KDF parameter.
        n_ab : int
            Number of second quadrature nodes.

        Returns
        -------
        xi_second : array
            Array of shape (n_ab, len(xi_first)) containing quadrature nodes.
        w_second : array
            Array of shape (n_ab, len(xi_first)) containing quadrature weights.

        """
        xi = np.add.outer(self.xi_hermite*sqrt(2)*sigma, xi_first)
        w = np.tile(self.w_hermite_norm, (np.size(xi_first),1)).T
        return xi, w

    def _sigma_bounds(self, mom):
        return (0., (1. + 1e-12)*self._sigma_analyt(mom[:3])) # Exact sigma_max may result in divisions by zero

    def _sigma_analyt(self, mom):
        """
        Analytical calculation of sigma if the number of first quadrature nodes N < 3. The analytical expression for N=2 was taken from Ref. [:cite:label:`Chalons_2010`].

        Parameters
        ----------
        mom : array
            Set of 2*N + 1 moments to calculate an N-node quadrature.

        Returns
        -------
        sigma : float or None
            Analytically calculated value of sigma if possible, `None` otherwise.

        References
        ----------
            +----------------+----------------------+
            | [Chalons_2010] | :cite:`Chalons_2010` |
            +----------------+----------------------+

        """
        if len(mom) == 3:
            return sqrt(mom[2]/mom[0] - (mom[1]/mom[0])**2)
        if len(mom) == 5:
            # Normalization so that m[0] = 1 (does not affect the result and simplifies the formulas)
            m = mom/mom[0]
            # Proposition 1 in Ref. [Chalons_2010]
            e = m[2] - m[1]**2
            q = m[3] - m[1]**3 - 3*m[1]*e
            eta = m[4] - 3*m[1]**4 - 4*m[3]*m[1] + 6*m[2]*m[1]**2
            sigma_0 = np.roots([2., 0., eta - 3*e**2, q**2])
            sigma_0 = float(sigma_0[np.isclose(sigma_0.imag, 0.)].real) # Real root is unique, see [Chalons_2010]
            return (sigma_0 + e)**0.5
        return None

    def ndf(self, xi, sigma_zero=0.):
        """
        Compute :math:`n(\\xi)`, in this case a weighted sum of normal densities

        .. math::

            n(\\xi) = \\frac{1}{\sqrt{2\pi} \sigma} \sum\limits_{j=1}^N w_j e^{-\\frac{1}{2} \\frac{(\\xi - \\xi_j)^2}{\sigma^2}},

        where :math:`\\xi_1 \dots \\xi_N` are the nodes of the first quadrature and :math:`\sigma` is the EQMOM parameter corresponding to the first :math:`2N+1` moments.

        Parameters
        ----------
        xi : array or float
            Abscissa.
        sigma_zero : float, optional
            Value to use if sigma=0. The default 0 causes an error.

        Returns
        -------
        ndf : array or float
            NDF-value(s) at xi.

        """
        sigma = self.sigma if self.sigma != 0. else sigma_zero
        ndf = 1/(2*np.pi)**0.5/sigma
        ndf *= np.exp(-0.5*((xi - self.xi_first)/sigma)**2)@self.w_first
        return ndf


class LaplaceEqmom(ExtendedQmom):
    """
    EQMOM with Laplace-KDFs as proposed by Pigou et al. [:cite:label:`Pigou_2018`].

    Parameters
    ----------
    kwargs :
        See class `ExtendedQmom`.

    Attributes
    ----------
    inversion : MomentInversion
        Basic moment inversion algorithm to compute quadrature from moments, see class `MomentInversion`.
    n_ab : int
        Number of second quadrature nodes per first quadrature node.
    atol : float
        Absolute tolerance used to find EQMOM-parameter sigma.
    rtol : float
        Relative tolerance used to find EQMOM-parameter sigma.
    n_init : int
        Initial number of first quadrature nodes, needed for some initializations.
    xi_first : array
        Abscissas of the first quadrature.
    xi_second : array
        Array with shape `(n_ab, len(xi_first))` containing abscissas of the second quadrature for each first quadrature node.
    w_second : array
        Array of shape `(n_ab, len(xi_first))` with weights of the second quadrature for each first quadrature node.
    sigma : float
        KDF-specific EQMOM parameter.
    A_coeffs : array
        Constant factors of sigma in A-matrix (map m* -> m), for details see [:cite:label:`Yuan_2012`]).
    Ainv_coeffs : array
        Constant factors of sigma in inverse of A (map m -> m*), for details see [:cite:label:`Yuan_2012`]).
    sigma_pow : array
        Sigma powers corresponding to elements in A and Ainv.
    xi_std : array
        Nodes of the 'standard' Laplace distribution to compute the second quadrature by simple linear transformation.
    w_std : array
        Weights of the 'standard' Laplace distribution to compute the second quadrature by simple linear transformation.

    References
    ----------
        +--------------+--------------------+
        | [Pigou_2018] | :cite:`Pigou_2018` |
        +--------------+--------------------+
        | [Yuan_2012]  | :cite:`Yuan_2012`  |
        +--------------+--------------------+

    """
    name = 'laplaceEQMOM'
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        nmom = 2*self.n_init + 1
        mom_std = laplace_moments(2*self.n_ab, mu=0., b=1.)
        self.xi_std, self.w_std = self.inversion(mom_std)

    @classmethod
    def new(cls, qbmm_setup, **kwargs):
        """
        Return instance of `LaplaceEqmom` class (necessary for the dynamic selection from parent class to work).

        Parameters
        ----------
        qbmm_setup : dict
            Setup dictionary containing required parameters, see base classes `UnivariateQbmm` and `ExtendedQmom`.

        Returns
        -------
        new : LaplaceEqmom
            New `LaplaceEqmom` object.

        """
        return cls(**qbmm_setup)

    def find_sigma(self, mom):
        """
        Routine to find the EQMOM parameter :math:`\sigma`, which is the root of the target function :math:`\mathbf{J}(\sigma)`, see [:cite:label:`Yuan_2012`]. For the EQMOM with Laplace-KDFs (Hamburger problem) the improved method based on moment realizability proposed in Ref. [:cite:label:`Pigou_2018`] is used.

        Parameters
        ----------
        mom : array
            Realizable set of moments.

        Returns
        -------
        sigma : float
            The found root of the target function.
        xi : array
            Nodes of the first quadrature.
        w : array
            Weights of the first quadrature.

        References
        ----------
            +--------------+--------------------+
            | [Yuan_2012]  | :cite:`Yuan_2012`  |
            +--------------+--------------------+
            | [Pigou_2018] | :cite:`Pigou_2018` |
            +--------------+--------------------+

        """
        bounds = self._sigma_bounds(mom)
        return eqroots.pigou_hamburger(bounds[1], bounds, mom, self.m2ms, self.ms2m, \
            self.inversion, self.atol, self.rtol)

    def _init_sigma_pow(self, nmom):
        """
        Initialize array containing the powers of :math:`\sigma` in the lower triangular matrices :math:`\mathbf{A}(\sigma)` and :math:`\mathbf{A}^{-1}(\sigma)` as described in Ref. [:cite:label:`Pigou_2018`] (Appendix B.2.2).

        Parameters
        ----------
        size : int
            Number of moments.

        Returns
        -------
        sigma_pow : array
            Powers of sigma with shape (nmom, nmom).

        References
        ----------
            +--------------+--------------------+
            | [Pigou_2018] | :cite:`Pigou_2018` |
            +--------------+--------------------+

        """
        # Compare Eqs. (B.12)-(B.15) in Ref. [Pigou_2018]
        sigma_pow = toeplitz([i*(1 - i%2) for i in range(nmom)], [0]*(nmom))
        return sigma_pow

    def _init_A_coeffs(self, size):
        """
        Initialize matrix with the constant factors in the lower triangular matrix :math:`\mathbf{A}(\sigma)` (map :math:`\mathbf{m}^{*} \\rightarrow \mathbf{m}`) as described in Ref. [:cite:label:`Pigou_2018`] (Appendix B.2.2).

        Parameters
        ----------
        size : int
            Size of square matrix A.

        Returns
        -------
        A_coeffs: array
            Constant factors in the matrix A with shape (size,size).

        References
        ----------
            +--------------+--------------------+
            | [Pigou_2018] | :cite:`Pigou_2018` |
            +--------------+--------------------+

        """
        A = np.identity(size)
        for i in range(size):
            for j in range(i%2,i,2):
                # Eq. (B.12) in Ref. [Pigou_2018]
                A[i,j] = factorial(i)/factorial(j)
        return A

    def _init_Ainv_coeffs(self, size):
        """
        Initialize matrix with the constant factors in the lower triangular matrix :math:`\mathbf{A}^{-1}(\sigma)` (map :math:`\mathbf{m} \\rightarrow \mathbf{m}^{*}`) as described in Ref. [:cite:label:`Pigou_2018`] (Appendix B.2.2).

        Parameters
        ----------
        size : int
            Size of square matrix Ainv.

        Returns
        -------
        Ainv_coeffs: array
            Constant factors in the matrix Ainv with shape (size,size).

        References
        ----------
            +--------------+--------------------+
            | [Pigou_2018] | :cite:`Pigou_2018` |
            +--------------+--------------------+

        """
        Ainv = np.identity(size)
        k_2 = np.arange(size-2)
        # Eq. (B.13) in Ref. [Pigou_2018]
        Ainv -= np.diag((k_2 + 1)*(k_2 + 2), k=-2)
        return Ainv

    def second_quad(self, xi_first, sigma, n_ab):
        """
        Compute second quadrature from the first quadrature and EQMOM-parameter :math:`\sigma`, which is, in the case of Laplace-KDFs, the scale parameter of a Laplace distribution. In this case, the second quadrature can be calculated by a simple linear transformation of the precomputed nodes and weights of a 'standard' Laplace distribution (location parameter 0 and scale parameter 1).

        Parameters
        ----------
        xi_first : array or float
            First quadrature nodes.
        sigma : float
            KDF parameter.
        n_ab : int
            Number of second quadrature nodes.

        Returns
        -------
        xi_second : array
            Array of shape (n_ab, len(xi_first)) containing quadrature nodes.
        w_second : array
            Array of shape (n_ab, len(xi_first)) containing quadrature weights.

        """
        xi = np.add.outer(self.xi_std*sigma, xi_first)
        w = np.tile(self.w_std, (np.size(xi_first),1)).T
        return xi, w

    def _sigma_bounds(self, mom):
        return (0., (1. + 1e-12)*self._sigma_analyt(mom[:3])) # exact sigma_max may result in divisions by zero

    def _sigma_analyt(self, mom):
        """
        Analytical calculation of sigma for a single node.

        Parameters
        ----------
        moments : array
            Set of 2*N + 1 moments to calculate an N-node quadrature.

        Returns
        -------
        sigma : float or None
            Analytically calculated value of sigma if possible, `None` otherwise.

        """
        if len(mom) < 4:
            sigma = 0.5*(mom[2]*mom[0] - mom[1]**2)
            return sigma**0.5/mom[0]
        return None

    def ndf(self, xi, sigma_zero=0.):
        """
        Compute :math:`n(\\xi)`, in this case a weighted sum of Laplace-densities

        .. math::

            n(\\xi) = \\frac{1}{2 \sigma} \sum\limits_{j=1}^N w_j e^{-\\frac{|\\xi - \\xi_j|}{\sigma}},

        where :math:`\\xi_1 \dots \\xi_N` are the nodes of the first quadrature and :math:`\sigma` is the EQMOM parameter corresponding to the first :math:`2N+1` moments.

        Parameters
        ----------
        xi : array or float
            Abscissa.
        sigma_zero : float, optional
            Value to use if sigma=0. The default 0 causes an error.

        Returns
        -------
        ndf : array or float
            NDF-value(s) at xi.

        """
        sigma = self.sigma if self.sigma != 0. else sigma_zero
        ndf = 0.5/sigma
        ndf *= np.exp(-abs(xi - self.xi_first)/sigma)@self.w_first
        return ndf


class GammaEqmom(ExtendedQmom):
    """
    EQMOM with Gamma-KDFs.

    Parameters
    ----------
    kwargs :
        See base class `ExtendedQmom`.

    Attributes
    ----------
    inversion : MomentInversion
        Basic moment inversion algorithm to compute quadrature from moments, see class `MomentInversion`.
    n_ab : int
        Number of second quadrature nodes per first quadrature node.
    atol : float
        Absolute tolerance used to find EQMOM-parameter sigma.
    rtol : float
        Relative tolerance used to find EQMOM-parameter sigma.
    n_init : int
        Initial number of first quadrature nodes, needed for some initializations.
    xi_first : array
        Abscissas of the first quadrature.
    xi_second : array
        Array with shape `(n_ab, len(xi_first))` containing abscissas of the second quadrature for each first quadrature node.
    w_second : array
        Array of shape `(n_ab, len(xi_first))` with weights of the second quadrature for each first quadrature node.
    sigma : float
        KDF-specific EQMOM parameter.
    A_coeffs : array
        Constant factors of sigma in A-matrix (map m* -> m), for details see [:cite:label:`Yuan_2012`]).
    Ainv_coeffs : array
        Constant factors of sigma in inverse of A (map m -> m*), for details see [:cite:label:`Yuan_2012`]).
    sigma_pow : array
        Sigma powers corresponding to elements in A and Ainv.

    References
    ----------
        +-------------+-------------------+
        | [Yuan_2012] | :cite:`Yuan_2012` |
        +-------------+-------------------+

    """
    name = 'gammaEQMOM'
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def new(cls, qbmm_setup, **kwargs):
        """
        Return instance of `GammaEqmom` class (necessary for the dynamic selection from parent class to work).

        Parameters
        ----------
        qbmm_setup : dict
            Setup dictionary containing required parameters, see base classes `UnivariateQbmm` and `ExtendedQmom`.

        Returns
        -------
        new : GammaEqmom
            New `GammaEqmom` object.

        """
        return cls(**qbmm_setup)

    def find_sigma(self, mom):
        """
        Routine to find the EQMOM parameter :math:`\sigma`, which is the root of the target function :math:`\mathbf{J}(\sigma)`, see [:cite:label:`Yuan_2012`]. For the EQMOM with Gamma-KDFs (Stieltjes problem) the improved method based on moment realizability proposed in Ref. [:cite:label:`Pigou_2018`] is used.

        Parameters
        ----------
        mom : array
            Realizable set of moments.

        Returns
        -------
        sigma : float
            The found root of the target function.
        xi : array
            Nodes of the first quadrature.
        w : array
            Weights of the first quadrature.

        References
        ----------
            +--------------+--------------------+
            | [Yuan_2012]  | :cite:`Yuan_2012`  |
            +--------------+--------------------+
            | [Pigou_2018] | :cite:`Pigou_2018` |
            +--------------+--------------------+

        """
        bounds = self._sigma_bounds(mom)
        return eqroots.pigou_stieltjes(bounds[1], bounds, mom, self.m2ms, self.ms2m, \
            self.inversion, self.atol, self.rtol)

    def _init_sigma_pow(self, nmom):
        """
        Initialize array containing the powers of :math:`\sigma` in the lower triangular matrices :math:`\mathbf{A}(\sigma)` and :math:`\mathbf{A}^{-1}(\sigma)` as described in Ref. [:cite:label:`Pigou_2018`] (Appendix B.4.2).

        Parameters
        ----------
        size : int
            Number of moments.

        Returns
        -------
        sigma_pow : array
            Powers of sigma with shape (nmom, nmom).

        References
        ----------
            +--------------+--------------------+
            | [Pigou_2018] | :cite:`Pigou_2018` |
            +--------------+--------------------+

        """
        # Compare Eqs. (B.28)-(B.31) in Ref. [Pigou_2018]
        sigma_pow = np.zeros((nmom, nmom))
        for j in range(1, nmom-1):
            sigma_pow[1:,1:] += np.diag(j*np.ones(nmom-j-1),-j)
        return sigma_pow

    def _init_A_coeffs(self, size):
        """
        Initialize matrix with the constant factors in the lower triangular matrix :math:`\mathbf{A}(\sigma)` (map :math:`\mathbf{m}^{*} \\rightarrow \mathbf{m}`) as described in Ref. [:cite:label:`Pigou_2018`] (section B.4.2).

        Parameters
        ----------
        size : int
            Size of square matrix A.

        Returns
        -------
        A_coeffs: array
            Constant factors in the matrix A with shape (size,size).

        References
        ----------
            +--------------+--------------------+
            | [Pigou_2018] | :cite:`Pigou_2018` |
            +--------------+--------------------+

        """
        A = np.eye(size)
        for i in range(1, size):
            for j in range(1, i):
                # Eq. (B.28) in Ref. [Pigou_2018]
                A[i,j] = A[i-1,j-1] + (i-1)*A[i-1,j]
        return A

    def _init_Ainv_coeffs(self, size):
        """
        Initialize matrix with the constant factors in the lower triangular matrix :math:`\mathbf{A}^{-1}(\sigma)` (map :math:`\mathbf{m} \\rightarrow \mathbf{m}^{*}`) as described in Ref. [:cite:label:`Pigou_2018`] (Appendix B.4.2).

        Parameters
        ----------
        size : int
            Size of square matrix Ainv.

        Returns
        -------
        Ainv_coeffs: array
            Constant factors in the matrix Ainv with shape (size,size).

        References
        ----------
            +--------------+--------------------+
            | [Pigou_2018] | :cite:`Pigou_2018` |
            +--------------+--------------------+

        """
        Ainv = np.eye(size)
        for i in range(1, size):
            for j in range(1, i):
                # Eq. (B.29) in Ref. [Pigou_2018]
                Ainv[i,j] = Ainv[i-1,j-1] - j*Ainv[i-1,j]
        return Ainv

    def second_quad(self, xi_first, sigma, n_ab):
        """
        Compute second quadrature from the first quadrature and EQMOM-parameter :math:`\sigma`. In the case of Gamma-KDFs, the second quadrature is related to the generalized Laguerre quadrature, see Ref. [:cite:label:`Yuan_2012`].

        Parameters
        ----------
        xi_first : array or float
            First quadrature nodes.
        sigma : float
            KDF parameter.
        n_ab : int
            Number of second quadrature nodes.

        Returns
        -------
        xi_second : array
            Array of shape (n_ab, len(xi_first)) containing quadrature nodes.
        w_second : array
            Array of shape (n_ab, len(xi_first)) containing quadrature weights.

        References
        ----------
            +-------------+-------------------+
            | [Yuan_2012] | :cite:`Yuan_2012` |
            +-------------+-------------------+

        """
        try:
            n = len(xi_first)
        except TypeError:
            xi, w, sum_ = roots_genlaguerre(n_ab, xi_first/sigma - 1, True)
            return xi*sigma, w/sum_
        xi = np.empty((n_ab, n))
        w = np.empty_like(xi)
        for j in range(n):
            xi[:,j], w[:,j], sum_ = roots_genlaguerre(n_ab, xi_first[j]/sigma - 1, True)
            w[:,j] /= sum_
        return xi*sigma, w

    def _sigma_bounds(self, mom):
        return (0., (1. + 1e-12)*self._sigma_analyt(mom[:3])) # exact sigma_max may result in divisions by zero

    def _sigma_analyt(self, mom):
        """
        Analytical calculation of sigma for a single node.

        Parameters
        ----------
        moments : array
            Set of 2*N + 1 moments to calculate an N-node quadrature.

        Returns
        -------
        sigma : float or None
            Analytically calculated value of sigma if possible, `None` otherwise.

        """
        if len(mom) < 4:
            sigma = mom[2]/mom[1] - mom[1]/mom[0]
            return sigma
        return None

    def ndf(self, xi, sigma_zero=0.):
        """
        Compute :math:`n(\\xi)`, in this case a weighted sum of Gamma-densities

        .. math::

            n(\\xi) = \sum\limits_{j=1}^N w_j \\frac{1}{\Gamma(\\xi_j/\sigma) \sigma^{\\xi_j/\sigma}}\\xi^{\\xi_j/\sigma - 1} e^{-\\xi/\sigma}

        where :math:`\Gamma` denotes the gamma function, :math:`\\xi_1 \dots \\xi_N` are the nodes of the first quadrature and :math:`\sigma` is the EQMOM parameter corresponding to the first :math:`2N+1` moments.

        Parameters
        ----------
        xi : array or float
            Abscissa.
        sigma_zero : float, optional
            Value to use if sigma=0. The default 0 causes an error.

        Returns
        -------
        ndf : array or float
            NDF-value(s) at xi.

        """
        sigma = self.sigma if self.sigma != 0. else sigma_zero
        lambda_ = self.xi_first/sigma
        ndf = xi**(lambda_ - 1) * np.exp(-xi/sigma)
        ndf /= gamma_func(lambda_)*sigma**lambda_
        ndf = ndf@self.w_first
        return ndf


class BetaEqmom(ExtendedQmom):
    """
    EQMOM with Beta-KDFs.

    Parameters
    ----------
    kwargs :
        See base class `ExtendedQmom`.

    Attributes
    ----------
    inversion : MomentInversion
        Basic moment inversion algorithm to compute quadrature from moments.
    n_ab : int
        Number of second quadrature nodes per first quadrature node.
    atol : float
        Absolute tolerance used to find EQMOM-parameter sigma.
    rtol : float
        Relative tolerance used to find EQMOM-parameter sigma.
    n_init : int
        Initial number of first quadrature nodes, needed for some initializations.
    xi_first : array
        Abscissas of the first quadrature.
    xi_second : array
        Array with shape `(n_ab, len(xi_first))` containing abscissas of the second quadrature for each first quadrature node.
    w_second : array
        Array of shape `(n_ab, len(xi_first))` with weights of the second quadrature for each first quadrature node.
    sigma : float
        KDF-specific EQMOM parameter.
    A_coeffs : array
        Constant factors of sigma in A-matrix (map m* -> m), for details see [:cite:label:`Yuan_2012`]).
    Ainv_coeffs : array
        Constant factors of sigma in inverse of A (map m -> m*), for details see [:cite:label:`Yuan_2012`]).
    sigma_pow : array
        Sigma powers corresponding to elements in A and Ainv.

    References
    ----------
        +-------------+-------------------+
        | [Yuan_2012] | :cite:`Yuan_2012` |
        +-------------+-------------------+

    """
    name = 'betaEQMOM'
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def new(cls, qbmm_setup, **kwargs):
        """
        Return instance of `BetaEqmom` class (necessary for the dynamic selection from parent class to work).

        Parameters
        ----------
        qbmm_setup : dict
            Setup dictionary containing required parameters, see base classes `UnivariateQbmm` `ExtendedQmom`.

        Returns
        -------
        new : BetaEqmom
            New `BetaEqmom` object.

        """
        return cls(**qbmm_setup)

    # TODO make Pigou-root-finding algorithm more robust
    def find_sigma(self, mom):
         bounds = self._sigma_bounds(mom)
         return eqroots.pigou_hausdorff(bounds[1], bounds, mom, self.m2ms, self.ms2m, \
            self.inversion, self.atol, self.rtol)
        #
        # return super().find_sigma(mom)

    def _init_sigma_pow(self, nmom):
        """
        Initialize array containing the powers of :math:`\sigma` in the lower triangular matrices :math:`\mathbf{A}(\sigma)` and :math:`\mathbf{A}^{-1}(\sigma)` as described in Ref. [:cite:label:`Pigou_2018`] (Appendix B.6.2).

        Parameters
        ----------
        size : int
            Number of moments.

        Returns
        -------
        sigma_pow : array
            Powers of sigma with shape (nmom, nmom).

        References
        ----------
            +--------------+--------------------+
            | [Pigou_2018] | :cite:`Pigou_2018` |
            +--------------+--------------------+

        """
        # Sigma powers in the numerator are equal to those of Gamma-EQMOM,
        # compare Eq. (B.49) in Ref. [Pigou_2018]. Denominator (Eq. (B.50))
        # is computed for every sigma-iteration.
        sigma_pow = np.zeros((nmom, nmom))
        for j in range(1, nmom-1):
            sigma_pow[1:,1:] += np.diag(j*np.ones(nmom-j-1),-j)
        return sigma_pow

    def _init_A_coeffs(self, size):
        """
        Initialize matrix with the constant factors in the lower triangular matrix :math:`\mathbf{A}(\sigma)` (map :math:`\mathbf{m}^{*} \\rightarrow \mathbf{m}`) as described in Ref. [:cite:label:`Pigou_2018`] (section B.6.2).

        Parameters
        ----------
        size : int
            Size of square matrix A.

        Returns
        -------
        A_coeffs: array
            Constant factors in the matrix A with shape (size,size).

        References
        ----------
            +--------------+--------------------+
            | [Pigou_2018] | :cite:`Pigou_2018` |
            +--------------+--------------------+

        """
        # Sigma-independent coefficients are equal to those of
        # Gamma-EQMOM, compare Eq. (B.49) in Ref. [Pigou_2018].
        A = np.eye(size)
        for i in range(1, size):
            for j in range(1, i):
                # Eq. (B.28) in Ref. [Pigou_2018]
                A[i,j] = A[i-1,j-1] + (i-1)*A[i-1,j]
        return A

    def _init_Ainv_coeffs(self, size):
        """
        Initialize matrix with the constant factors in the lower triangular matrix :math:`\mathbf{A}^{-1}(\sigma)` (map :math:`\mathbf{m} \\rightarrow \mathbf{m}^{*}`) as described in Ref. [:cite:label:`Pigou_2018`] (Appendix B.6.2).

        Parameters
        ----------
        size : int
            Size of square matrix Ainv.

        Returns
        -------
        Ainv_coeffs: array
            Constant factors in the matrix Ainv with shape (size,size).

        References
        ----------
            +--------------+--------------------+
            | [Pigou_2018] | :cite:`Pigou_2018` |
            +--------------+--------------------+

        """
        # Sigma-independent coefficients are equal to those of
        # Gamma-EQMOM, compare Eq. (B.51) in Ref. [Pigou_2018].
        Ainv = np.eye(size)
        for i in range(1, size):
            for j in range(1, i):
                # Eq. (B.29) in Ref. [Pigou_2018]
                Ainv[i,j] = Ainv[i-1,j-1] - j*Ainv[i-1,j]
        return Ainv

    def A(self, sigma, nmom):
        """
        Assemble Matrix :math:`\mathbf{A}(\sigma)`, the linear map :math:`\mathbf{m}^{*} \\rightarrow \mathbf{m}`, for Beta-EQMOM as described in [:cite:label:`Pigou_2018`] (Appendix B.6.2).

        Parameters
        ----------
        sigma : float
            KDF parameter.
        nmom : int
            Number of moments

        Returns
        -------
        A : array
            2D-array representing the matrix `A`.

        References
        ----------
            +--------------+--------------------+
            | [Pigou_2018] | :cite:`Pigou_2018` |
            +--------------+--------------------+

        """
        # See Eqs. (B.49) and (B.50) in Ref. [Pigou_2018]
        f = 1.
        A_ = self.A_coeffs[:nmom,:nmom]*sigma**self.sigma_pow[:nmom,:nmom]
        for i in range(2, nmom):
            f *= 1 + (i-1)*sigma
            A_[i,:i+1] /= f
        return A_

    def Ainv(self, sigma, nmom):
        """
        Assemble Matrix :math:`\mathbf{A}^{-1}(\sigma)`, the linear map :math:`\mathbf{m} \\rightarrow \mathbf{m}^{*}`, for Beta-EQMOM as described in [:cite:label:`Pigou_2018`] (Appendix B.6.2).

        Parameters
        ----------
        sigma : float
            KDF parameter.
        nmom : int
            Number of moments

        Returns
        -------
        Ainv : array
            2D-array representing the matrix `Ainv`.

        References
        ----------
            +--------------+--------------------+
            | [Pigou_2018] | :cite:`Pigou_2018` |
            +--------------+--------------------+

        """
        # See Eqs. (B.50) and (B.51) in Ref. [Pigou_2018]
        f = 1.
        Ainv_ = self.Ainv_coeffs[:nmom,:nmom]*sigma**self.sigma_pow[:nmom,:nmom]
        for j in range(2, nmom):
            f *= 1 + (j-1)*sigma
            Ainv_[:,j] *= f
        return Ainv_

    def second_quad(self, xi_first, sigma, n_ab):
        """
        Compute second quadrature from the first quadrature and EQMOM-parameter :math:`\sigma`. In the case of Beta-KDFs, the second quadrature is related to the Gauss-Jacobi quadrature, see [:cite:label:`Yuan_2012`].

        Parameters
        ----------
        xi_first : array or float
            First quadrature nodes.
        sigma : float
            KDF parameter.
        n_ab : int
            Number of second quadrature nodes.

        Returns
        -------
        xi_second : array
            Array of shape (n_ab, len(xi_first)) containing quadrature nodes.
        w_second : array
            Array of shape (n_ab, len(xi_first)) containing quadrature weights.

        References
        ----------
            +-------------+-------------------+
            | [Yuan_2012] | :cite:`Yuan_2012` |
            +-------------+-------------------+

        """
        # Parameters of the Jacobi polynomials
        alpha = (1 - xi_first)/sigma - 1
        beta = xi_first/sigma - 1
        try:
            n = len(xi_first)
        except TypeError:
            xi, w, sum_ = roots_jacobi(n_ab, alpha, beta, True)
            return 0.5*(xi + 1), w/sum_
        xi = np.empty((n_ab, n))
        w = np.empty_like(xi)
        for j in range(n):
            xi[:,j], w[:,j], sum_ = roots_jacobi(n_ab, alpha[j], beta[j], True)
            w[:,j] /= sum_

        # Return transformed variable: [-1, 1] -> [0, 1]
        return 0.5*(xi + 1), w

    def _sigma_bounds(self, mom):
        return (0., (1. + 1e-12)*self._sigma_analyt(mom[:3])) # exact sigma_max may result in divisions by zero

    def _sigma_analyt(self, mom):
        """
        Analytical calculation of sigma for a single node.

        Parameters
        ----------
        moments : array
            Set of 2*N + 1 moments to calculate an N-node quadrature.

        Returns
        -------
        sigma : float or None
            Analytically calculated value of sigma if possible, `None` otherwise.

        """
        if len(mom) < 4:
            sigma = (mom[1]**2 - mom[0]*mom[2]) / (mom[0]*(mom[2] - mom[1]))
            return sigma
        return None

    def ndf(self, xi, sigma_zero=0.):
        """
        Compute :math:`n(\\xi)`, in this case a weighted sum of normal densities

        .. math::

            n(\\xi) = \\sum\\limits_{j=1}^{N} w_j \\frac{\\xi^{\\xi_j/\\sigma} (1-\\xi)^{(1-\\xi_j)/\\sigma - 1}}{B(\\xi_j/\\sigma, (1 - \\xi_j)/\\sigma)},

        where :math:`B` is the beta function, :math:`\\xi_1 \dots \\xi_N` are the nodes of the first quadrature and :math:`\sigma` is the EQMOM parameter corresponding to the first :math:`2N+1` moments.

        Parameters
        ----------
        xi : array or float
            Abscissa.
        sigma_zero : float, optional
            Value to use if sigma=0. The default 0 causes an error.

        Returns
        -------
        ndf : array or float
            NDF-value(s) at xi.

        """
        sigma = self.sigma if self.sigma != 0. else sigma_zero
        lambda_ = self.xi_first/sigma
        mu = (1 - self.xi_first)/sigma
        ndf = xi**(lambda_ - 1) * (1 - xi)**(mu - 1)
        ndf /= beta_func(lambda_, mu)
        ndf = ndf@self.w_first
        return ndf
