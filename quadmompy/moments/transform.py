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

import numpy as np
from scipy.special import binom
from quadmompy.core.hankel import hankel_det


def linear_transform(mom, a, b):
    """
    Linear moment transformation, i.e. computation of the moments of a random variable t = a*x + b given the moments of x.

    Parameters
    ----------
    mom : array
        Original moment set corresponding to the random variable x.
    a : float
        Scaling factor.
    b : float
        Shift.

    Returns
    -------
    mu : array
        Transformed moment set.

    """
    mu = [sum([binom(k, j)*a**(k-j)*b**j*mom[k-j] for j in range(k+1)]) for k in range(len(mom))]
    return np.array(mu)


def mom2canonmom(mom):
    """
    Transformation of a Hausdorff moment sequence (support [0,1]) to canonical moments as reported in Ref. [:cite:label:`Dette_1997`].

    Parameters
    ----------
    mom : array
        A valid Hausdorff moment sequence.

    Returns
    -------
    p : array
        Canonical moments corresponding to the given moment sequence.

    Notes
    ----
    It should be more efficient to use another algorithm, e.g. the Q-D-algorithm, to compute the continued-fraction coefficients zeta and subsequently the canonical moments instead of using Hankel determinants.

    References
    ----------
        +---------------+--------------------+
        | [Dette_1997]  | :cite:`Dette_1997` |
        +---------------+--------------------+

    """
    nmom = len(mom)
    p = mom.copy()

    # Compute required Hankel determinants, Eq. (1.4.3) in Ref. [Dette_1997]
    h_bottom = np.ones(nmom+1)
    h_top = np.ones(nmom+1)
    for k in range(1, nmom):
        h_bottom[k] = hankel_det(mom[:k+1], 'upper')
        h_top[k] = hankel_det(mom[:k] - mom[1:k+1], 'upper')

    # Canonical moments from Hankel determinants
    p[1] /= p[0]
    p[2:] = h_bottom[2:-1]*h_top[:-3]/(h_bottom[1:-2]*h_top[1:-2])      # corollary 1.4.6 in Ref. [Dette_1997]
    return p


def canonmom2mom(p, m0=1.):
    """
    Transformation of a sequence of canonical moments to the corresponding Hausdorff moment sequence (support [0,1]).

    Parameters
    ----------
    p : array
        A valid Hausdorff moment sequence.
    m0 : float, optional
        Zeroth ordinary moment.

    Returns
    -------
    mom : array
        Ordinary moments corresponding to the given canonical moments.

    """
    # Compute recurrence coefficients of orthogonal polynomials
    alpha, beta = canonmom2rc(p)

    # Compute moments from recurrence coefficients
    beta[0] = m0
    mom = rc2mom(alpha, beta)
    return mom


def rc2ops(alpha, beta):
    """
    Compute orthogonal polynomial system given the recurrence coefficients.

    Parameters
    ----------
    alpha : array
        First set of recurrence coefficients.
    beta : array
        Second set of recurrence coefficients.

    Returns
    -------
    ops : array
        Lower left triangular matrix with the nth row containing the coefficients of the monic nth-degree polynomial corresponing to the powers in ascending order.

    """
    n = len(alpha)
    ops = np.zeros((n + 1, n + 1))
    ops[0,0] = 1.
    for k in range(n):
        ops[k+1,:k+2] = np.polymul([-alpha[k], 1], ops[k,:k+1])  - beta[k]*ops[k-1,:k+2]
    return ops


def rc2mom(alpha, beta):
    """
    Compute set of moments given the associated recurrence coefficients of orthogonal polynomials.

    Parameters
    ----------
    alpha : array
        First set of recurrence coefficients.
    beta : array
        Second set of recurrence coefficients.

    Returns
    -------
    mom : array
        Set of moments.

    Notes
    ----
    This is a quick preliminary solution. The ordinary moments should be calculated more efficiently from the recurrence coefficients by avoiding the explicit computation of orthogonal polynomials.

    References
    ----------
        +------------------+-----------------------+
        | [Simon_1998]     | :cite:`Simon_1998`    |
        +------------------+-----------------------+

    """
    nmom = len(alpha) + len(beta)
    # Assemble Jacobi matrix
    n = len(beta)
    J = np.diag(beta[1:]**0.5, -1)
    J += J.T
    if len(alpha) == n:
        J += np.diag(alpha)
    else:
        # (n-1,n-1)th element has no effect on
        # the first 2n-1 moments and is thus
        # arbitrary (here it is set to 1)
        J += np.diag(np.append(alpha, 1))

    # The kth moment is the top-left element of the kth
    # matrix power of J (see Ref. [Simon_1998], p. 93)
    Jpow = np.eye(n, dtype=beta.dtype)
    mom = np.ones(nmom, beta.dtype)
    mom[0] = beta[0]
    for k in range(nmom):
        mom[k] *= Jpow[0,0]
        Jpow = Jpow@J

    return mom


def zeta2rc(zeta):
    """
    Compute the recurrence coefficients of the orthogonal polynomials associated with a measure given the coefficients of its Stieltjes-transform's continued-fraction expansion.

    Parameters
    ----------
    zeta : array
        Continued fraction coefficients.

    Returns
    -------
    alpha : array
        First set of recurrence coefficients.
    beta : array
        Second set of recurrence coefficients.

    References
    ----------
        +--------------+--------------------+
        | [Dette_1997] | :cite:`Dette_1997` |
        +--------------+--------------------+

    """
    # follows from Eq. (1.7.3) in Ref. [Dette_1997]
    n = len(zeta)
    alpha = np.zeros(n//2)
    alpha[0] = zeta[1]
    alpha[1:] = zeta[2:-1:2] + zeta[3::2]
    beta = np.ones((n + 1)//2)
    beta[1:] = zeta[1:-1:2]*zeta[2::2]

    return alpha, beta


def rc2zeta(alpha, beta):
    """
    Compute the continued-fraction expansion coefficients of the Stieltjes transform of a measure given the recurrence coefficients of the associated orthogonal polynomials.

    Parameters
    ----------
    alpha : array
        First set of recurrence coefficients.
    beta : array
        Second set of recurrence coefficients.

    Returns
    -------
    zeta : array
        Continued-fraction coefficients.

    References
    ----------
        +--------------+--------------------+
        | [Dette_1997] | :cite:`Dette_1997` |
        +--------------+--------------------+

    """
    nmom = len(alpha) + len(beta)
    iodd = nmom % 2

    # follows from Eq. (1.7.3) in Ref. [Dette_1997]
    zeta = np.zeros(nmom)
    zeta[0] = 1
    zeta[1] = alpha[0]
    for k in range(1,nmom//2):
        if zeta[2*k-1] == 0:
            return zeta
        zeta[2*k] = beta[k]/zeta[2*k-1]
        zeta[2*k+1] = alpha[k] - zeta[2*k]
    if iodd and zeta[-2] != 0:
        zeta[-1] = beta[-1]/zeta[-2]

    return zeta


def canonmom2zeta(p):
    """
    Compute canonical moments from the continued-fraction coefficients associated with a measure with support [0,1].

    Parameters
    ----------
    p : array
        Set of canonical moments.

    Returns
    -------
    alpha : array
        First set of recurrence coefficients.
    beta : array
        Second set of recurrence coefficients.

    References
    ----------
        +--------------+--------------------+
        | [Dette_1997] | :cite:`Dette_1997` |
        +--------------+--------------------+

    """
    # Eq. (1.4.6) in Ref. [Dette_1997]
    q = 1 - p
    zeta = p.copy()
    zeta[2:] = p[2:]*q[1:-1]

    return zeta


def zeta2canonmom(zeta):
    """
    Compute the canonical moments associated to a measure with support [0,1] given the coefficients of its Stieltjes-transform's continued-fraction expansion.

    Parameters
    ----------
    zeta : array
        Continued-fraction coefficients.

    Returns
    -------
    p : array
        Canonical moments.

    References
    ----------
        +--------------+--------------------+
        | [Dette_1997] | :cite:`Dette_1997` |
        +--------------+--------------------+

    """
    # Eq. (1.4.6) in Ref. [Dette_1997]
    p = np.zeros_like(zeta)
    for k in range(1, len(p)):
        p[k] = zeta[k]/(1 - p[k-1])
        if p[k] == 0:
            break
    p[0] = 1.

    return p


def canonmom2rc(p):
    """
    Compute canonical moments from the recurrence coefficients of the orthogonal polynomials associated with a measure with support [0,1].

    Parameters
    ----------
    p : array
        Set of canonical moments.

    Returns
    -------
    alpha : array
        First set of recurrence coefficients.
    beta : array
        Second set of recurrence coefficients.

    """
    # canonical moments p -> continued-fraction coefficients zeta
    zeta = canonmom2zeta(p)
    # continued-fraction coefficients -> recurrence coefficients [alpha, beta]
    alpha, beta = zeta2rc(zeta)

    return alpha, beta


def rc2canonmom(alpha, beta):
    """
    Compute canonical moments from the recurrence coefficients of orthogonal polynomials associated with a Hausdorff moment sequence (support [0,1]).

    Parameters
    ----------
    alpha : array
        First set of recurrence coefficients.
    beta : array
        Second set of recurrence coefficients.

    Returns
    -------
    p : array
        Canonical moments.

    """
    # recurrence coefficients [alpha, beta] -> continued-fraction coefficients zeta
    zeta = rc2zeta(alpha, beta)
    # recurrence coefficients zeta -> canonical moments p
    p = zeta2canonmom(zeta)

    return p
