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
This module contains the basic moment inversion class that all the specific inversion algorithms are derived from.

"""
import numpy as np
import abc
from quadmompy.core import utils


class MomentInversion(metaclass=abc.ABCMeta):
    """
    Base class of moment inversion algorithms used to instantiate a selected subclass.

    This class is the base class for algorithms to compute Gaussian quadratures from moment sequences. It also serves as an interface to dynamically create instances of subclasses specified by name in a given setup dictionary.

    Parameters
    ----------
    radau : bool, optional
        Indicates if Gauss-Radau-quadrature (one fixed quadrature node) is to be computed. Default is `False`. If `radau==True`, the parameter `radau_node` must be provided as well.
    radau_node : float, optional
        Location of the fixed node of a Gauss-Radau quadrature (necessary if `radau==True`)
    lobatto : bool, optional
        Indicates if Gauss-Lobatto-quadrature (two fixed quadrature nodes) is to be computed. Default is `False`. If `lobatto==True`, the parameter `lobatto_nodes` must be provided as well.
    lobatto_nodes : array_like, optional
        Locations of the two fixed quadrature nodes of a Gauss-lobatto quadrature (necessary if `lobatto==True`)
    beta_tol : float
        Specifies the value below which the absolute value of a recurrence coefficient `beta` is considered to be zero.

    Attributes
    ----------
    radau : bool, optional
        Indicates if Gauss-Radau-quadrature (one fixed quadrature node) is to be computed. Default is `False`. If `radau==True`, the parameter `radau_node` must be provided as well.
    radau_node : float, optional
        Location of the fixed node of a Gauss-Radau quadrature (necessary if `radau==True`)
    lobatto : bool, optional
        Indicates if Gauss-Lobatto-quadrature (two fixed quadrature nodes) is to be computed. Default is `False`. If `lobatto==True`, the parameter `lobatto_nodes` must be provided as well.
    lobatto_nodes : array_like, optional
        Locations of the two fixed quadrature nodes of a Gauss-lobatto quadrature (necessary if `lobatto==True`)
    beta_tol : float
        Specifies the value below which the absolute value of a recurrence coefficient `beta` is considered to be zero.

    """
    def __init__(self, radau=False, radau_node=None, \
            lobatto=False, lobatto_nodes=None, beta_tol=1e-12, **kwargs):
        self.radau = radau
        self.radau_node = radau_node
        self.lobatto = lobatto
        self.lobatto_nodes = lobatto_nodes
        self. beta_tol = beta_tol
        if self.radau and self.lobatto:
            msg = "Conflicting parameters: Only one of `radau` and `lobatto` may be true."
            raise ValueError(msg)

    @classmethod
    def get_all_algorithms(cls):
        """
        Get list of all available inversion algorithms, i.e. all subclasses in the entire class hierarchy.

        Returns
        -------
        all_subclasses : list
            List of all types inheriting directly or indirectly from `MomentInversion`.

        """
        return utils.get_all_subclasses(cls)

    @classmethod
    def new(cls, name, adaptive=False, **kwargs):
        """
        Create an instance of a subclass given its name.

	Parameters
	----------
	name : str
	    The name of the specific moment inversion method.
        adaptive : bool, optional
            Specifies if the adaptive version of the algorithm is to be used. Default is `False`.
        kwargs :
            Parameters required by the inversion algorithm.

        Returns
        -------
        new : MomentInversion subclass
            Instance of the selected class.

        """
        algorithms = {scls.__name__: scls for scls in cls.get_all_algorithms()}
        if not name in algorithms.keys():
            err_message = "Unknown moment inversion algorithm '" + name + "'. Available algorithms: " + str([key for key in algorithms])
            raise ValueError(err_message)
        if name.find('Adaptive') == - 1 and adaptive:
            return algorithms[name + 'Adaptive'](**kwargs)
        return algorithms[name](**kwargs)

    @abc.abstractmethod
    def _compute_rc(self, mom, n, iodd, alpha, beta):
        """
        Compute recurrence coefficients of orthogonal polynomials associated with a given set of moments and assign them directly to the input arrays `alpha` and `beta`.

        Parameters
        ----------
        mom : array
            Set of realizable moments.
        n : int
            Number of recurrence coefficients to be computed.
        iodd : int
            Integer that takes the value 0 if the number of moments is even and 1 if it is odd.
        alpha : array
            Array to store the first set of recurrence coefficients that satisfies `len(alpha) >= n`.
        beta : array
            Array to store the second set of recurrence coefficients that satisfies `len(beta) >= n`.

        """
        pass

    def recurrence_coeffs(self, mom):
        """
        Compute recurrence coefficients of orthogonal polynomials corresponding to a given set of moments.

        Parameters
        ----------
        mom : array
            Set of realizable moments.

        Returns
        -------
        alpha : array
            First set of recurrence coefficients.
        beta : array
            Second set of recurrence coefficients.

        """
        nmom = len(mom)
        iodd = nmom % 2
        nmax = nmom//2 + 1

        # Initialize recurrence coefficients
        alpha = np.zeros(nmax)
        alpha[0] = mom[1]/mom[0]
        beta = np.zeros(nmax)
        beta[0] = mom[0]

        # Recurrence coefficients corresponding to Gauss-Radau quadrature if needed
        # (special case due to odd number of moments)
        if self.radau:
            self._compute_rc(mom[:(nmom + iodd - 1)], nmax, iodd, alpha, beta)
            return self._radau_correct(alpha, beta)

        n = nmax - 1 + iodd
        self._compute_rc(mom, nmax-1+iodd, iodd, alpha[:nmax-1+iodd], beta[:nmax-1+iodd])

        # Recurrence coefficients corresponding to Gauss-Lobatto quadrature if needed
        if self.lobatto:
            return self._lobatto_correct(alpha, beta)

        # Recurrence coefficients corresponding to regular Gauss quadrature
        return alpha[:-1], beta[:nmax-1+iodd]


    def quad_from_rc(self, alpha, beta):
        """
        Compute Gaussian quadrature abscissas and weights from recurrence coefficients of orthogonal polynomials, see e.g. [:cite:label:`Gautschi_2004`].

        Parameters
        ----------
        alpha : array
            First set of recurrence coefficients.
        beta : array
            Second set of recurrence coefficients.

        Returns
        -------
        xi : array
            Abscissae of the quadrature of the specified type.
        w : array
            Weights of the quadrature of the specified type.

        References
        ----------
            +-----------------+-----------------------+
            | [Gautschi_2004] | :cite:`Gautschi_2004` |
            +-----------------+-----------------------+

        """
        # Assemble tridiagonal Jacobi matrix
        jacobi = np.diag(alpha)
        for i in range(1,len(alpha)):
            jacobi[i,i-1] = beta[i]**0.5

        # The quadrature nodes are the eigenvalues of the Jacobi matrix and the
        # corresponding normalized weights are computed from the eigenvectors.
        xi, w = np.linalg.eigh(jacobi, 'L')
        w = w[0,:]**2
        return xi, w*beta[0]


    def moment_inversion(self, mom):
        """
        Computation of Gaussian quadrature from a given moment set.

        Parameters
        ----------
        mom : array
            Set of realizable moments. Depending on the setup, not all moments might be used.

        Returns
        -------
        xi : array
            Abscissae of the quadrature of the specified type (Gauss, Gauss-Radau, ...) corresponding to the given set of moments.
        w : array
            Weights of the quadrature of the specified type (Gauss, Gauss-Radau, ...) corresponding to the given set of moments.

        """
        n_mom = len(mom)
        a, b = self.recurrence_coeffs(mom)
        b[abs(b) < self.beta_tol] = 0.
        xi, w = self.quad_from_rc(a, b)
        return xi, w

    def _moment_inversion_ad(self, mom):
        """
        Moment inversion using adaptive procedure based on node-distance and weight-ratio criteria, called internally by adaptive algorithms.

        Parameters
        ----------
        mom : array_like
            The set of moments to be inverted.

        Returns
        -------
        xi : array_like
            Nodes of Gaussian quadrature.
        w : array_like
            Weights of Gaussian quadrature.

        """
        # Set initial values such that criteria are violated
        w_ratio = 0.5*self.rmin
        dab_ratio = self.eabs + 1
        nmom = len(mom)

        # Special case of only a single quadrature node
        n_nodes = nmom//2
        if n_nodes == 1:
            return mom[1]/mom[0]*np.ones(n_nodes), mom[0]*np.ones(n_nodes)

        # Initialize arrays using initial number of nodes
        xi = np.zeros(n_nodes)
        w = np.zeros(n_nodes)
        alpha = np.zeros(n_nodes)
        beta = np.zeros(n_nodes)

        # Check weight and node criteria and possibly reduce number of nodes
        while w_ratio < self.rmin or dab_ratio < self.eabs:
            xi[:] = 0.
            w[:] = 0.
            alpha[:n_nodes], beta[:n_nodes] = self.recurrence_coeffs(mom[:nmom])
            xi[:n_nodes], w[:n_nodes] = self.quad_from_rc(alpha[:n_nodes], beta[:n_nodes])
            if n_nodes == 1:
                break

            # Weight criterion
            w_ratio = min(w[:n_nodes])/max(w[:n_nodes])

            # Node criterion
            mindab = min(np.diff(xi[:n_nodes]))
            maxdab = max(np.diff(xi[:n_nodes]))
            dab_ratio = mindab/maxdab
            nmom -= 2
            n_nodes -= 1

        return xi, w

    def __call__(self, mom):
        """
        Computation of Gaussian quadrature from a given moment set, see method :meth:'moment_inversion' for more details.

        """
        return self.moment_inversion(mom)

    def _radau_correct(self, alpha, beta):
        """
        Correct recurrence coefficient :math:`\\alpha_{n-1}` for Gauss-Radau quadrature according to Ref. [:cite:label:`Gautschi_2004`].

        Parameters
        ----------
        alpha : array
            First set of recurrence coefficients of which the (n-1)th element is modified for Gauss-Radau quadrature. If len(alpha) < n, the array is extended with the computed coefficient.
        beta : array
            Second set of recurrence coefficient that remains unchanged.

        Returns
        -------
        alpha : array
            Modified first set of recurrence coefficients.
        beta : array
            Second set of recurrence coefficients (equals the input array).

        References
        ----------
            +-----------------+-----------------------+
            | [Gautschi_2004] | :cite:`Gautschi_2004` |
            +-----------------+-----------------------+

        """
        n = len(beta)
        if len(alpha) < n:
            alpha = np.append(alpha, 0.)

        # Compute orthogonal polynomials associated with given recurrence
        # coefficients evaluated at fixed Radau-node a.
        p_k = 0.
        p_kp1 = 1.
        a = self.radau_node
        for k in range(n-1):
            p_km1 = p_k
            p_k = p_kp1
            p_kp1 = (a - alpha[k])*p_k - beta[k]*p_km1

        #Correct recurrence coefficient, see Eq. (3.1.14) in Ref. [Gautschi_2004]
        alpha[n-1] = a - beta[n-1]*p_k/p_kp1

        return alpha, beta

    def _lobatto_correct(self, alpha, beta, extend=False):
        """
        Correct (or append) (n-1)th recurrence coefficients :math:`\\alpha_{n-1}` and :math:`\\beta_{n-1}` for Gauss-Lobatto quadrature according to Ref. [:cite:label:`Gautschi_2004`].

        Parameters
        ----------
        alpha : array
            First set of recurrence coefficients of which the (n-1)th element is modified for Gauss-Lobatto quadrature. If len(alpha) < n, the array is extended with the computed coefficient.
        beta : array
            Second set of recurrence coefficients of which the (n-1)th element is modified for Gauss-Lobatto quadrature. If len(beta) < n, the array is extended with the computed coefficient.
        extend : bool, optional
            If `True`, the modified Lobatto-recurrence coefficients are appended to alpha and beta. Otherwise, the last element is overwritten. The default value is `False`.

        Returns
        -------
        alpha : array
            Modified first set of recurrence coefficients.
        beta : array
            Modified second set of recurrence coefficients.

        References
        ----------
            +-----------------+-----------------------+
            | [Gautschi_2004] | :cite:`Gautschi_2004` |
            +-----------------+-----------------------+

        """
        n = len(beta)
        if extend:
            alpha = np.append(alpha, 0.)
            beta = np.append(beta, 0.)
            n += 1

        # Compute orthogonal polynomials associated with given recurrence
        # coefficients evaluated at Lobatto-nodes, to obtain the linear
        # system in (3.1.28), Ref. [Gautschi_2004]
        At = np.zeros((2,2))
        # References to matrix rows for readability
        p_k = At[1]
        p_kp1 = At[0]
        p_kp1[:] = 1.
        p_km1 = np.zeros_like(p_k)
        ab = self.lobatto_nodes
        #for k in range(n-1):
        for k in range(n-1):
            for j in range(2):
                p_km1[j] = p_k[j]
                p_k[j] = p_kp1[j]
                p_kp1[j] = (ab[j] - alpha[k])*p_k[j] - beta[k]*p_km1[j]

        # Correct recurrence coefficients, see Eq. (3.1.28) in Ref. [Gautschi_2004]
        alpha[-1], beta[-1] = np.linalg.solve(At.T, ab*At[0])

        return alpha, beta


# for backward-compatibility
BasicMomentInversion = MomentInversion
