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
This module contains (ordinary, absolute, canonical) moments of special weight functions / distributions.

"""
import numpy as np
from quadmompy.moments.transform import linear_transform
from scipy.special import factorial, factorial2, hyp1f1
from scipy.special import gamma as gamma_func


def beta_moments(n, alpha, beta, canonical=False):
    """
    Ordinary/canonical moments of the Beta-distribution with density

    .. math::

        f(x) = \\frac{\Gamma(\\alpha + \\beta) x^{\\alpha - 1} (1-x)^{\\beta - 1}}
                      {\Gamma(\\alpha) \Gamma(\\beta)},

    where :math:`\Gamma` is the gamma function.

    Parameters
    ----------
    n : int
        Number of moments.
    alpha : float
        First parameter of the Beta distribution, must be positive.
    beta : float
        Second parameter of the Beta distribution, must be positive.
    canonical : bool, optional
        Indicates if the canonical moments shall be computed (default: False).

    Returns
    -------
    mom : array
        Array of ordinary or canonical moments up to the (n-1)th order.

    References
    ----------
        +---------------+--------------------+
        | [Dette_1997]  | :cite:`Dette_1997` |
        +---------------+--------------------+
        | [Forbes_2011] | :cite:`Forbes_2011`|
        +---------------+--------------------+

    """
    # Ordinary moments, see Ref. [Forbes_2011]
    if not canonical:
        mom = np.ones(n)
        rec_num = alpha + np.arange(n)
        rec_coeffs = rec_num/(rec_num + beta)
        for k in range(1, n):
            mom[k] = rec_coeffs[k-1]*mom[k-1]
        return mom

    # Canonical moments, see Ref. [Dette_1997], Eq. (1.3.11)
    # Switch to parametrization used in Ref. [Dette_1997], see Eq. (1.3.9)
    beta, alpha = alpha - 1, beta - 1
    p = np.ones(n)
    k = np.arange(n + 1)
    p[2::2] = 0.5*k[2:-1:2]/(k[2:-1:2] + 1 + alpha + beta)
    p[1::2] = (0.5*k[2::2] + beta)/(k[2::2] + alpha + beta)
    return p


def uniform_moments(n, a=0., b=1., canonical=False):
    """
    Ordinary/canonical Moments of a random variable uniformly distributed between a and b.

    Parameters
    ----------
    n : int
        Number of moments.
    a : float, optional
        Lower limit, 0 by default.
    b : float
        Upper limit, 1 by default.
    canonical : bool, optional
        Indicates if the canonical moments shall be computed (default: False).

    Returns
    -------
    mom : array
        Array of ordinary or canonical moments up to the (n-1)th order.

    References
    ----------
        +---------------+--------------------+
        | [Dette_1997]  | :cite:`Dette_1997` |
        +---------------+--------------------+

    """
    # Ordinary moments
    if not canonical:
        kp1 = np.arange(n) + 1
        return (b**kp1 - a**kp1)/kp1/(b-a)

    # Canonical moments, see Ref. [Dette_1997], Example 1.4.11
    p = np.ones(n)
    p[1::2] = 0.5
    m = np.arange(1, (n + 1)//2)
    p[2::2] = m/(2*m + 1)
    return p


def normal_moments(n, mu, sigma, central=False, absolute=False):
    """
    Moments (raw/central/absolute) of a normal distribution.

    Parameters
    ----------
    n : int
        Number of moments.
    mu : float
        Mean of the distribution, must be real.
    sigma : float
        Standard deviation of the distribution, must be positive.
    central : bool, optional
        Specifies whether the central moments, i.e. the moments about the mean shall be computed, False by default.
    central : bool, optional
        Specifies whether the absolute moments shall be computed, False by default.

    Returns
    -------
    mom : array
        Array of moments of the specified kind up to order n-1.

    References
    ----------
        +--------------------+--------------------------+
        | [Winkelbauer_2012] | :cite:`Winkelbauer_2012` |
        +--------------------+--------------------------+

    """
    k = np.arange(n)
    # Raw moments
    if not absolute:
        # The formula for raw moments about the origin is nasty, see Ref. [Winkelbauer_2012].
        # So the non-central moments are computed by calculation of the central moments and
        # subsequent linear moment transformation.
        mom = np.zeros(n)
        mom[::2] = sigma**k[::2]*factorial2(k[::2] - 1)
        if central:
            return mom
        return linear_transform(mom, 1, mu)

    # Absolute moments
    if central:
        # Central absolute moments, see Ref. [Winkelbauer_2012]
        return sigma**k * (2**k/np.pi)**0.5 * gamma_func((k + 1)*0.5)
    # Raw absolute moments, see Ref. [Winkelbauer_2012]
    return sigma**k * 2**(0.5*k)/np.pi**0.5 * gamma_func((k + 1)*0.5) * hyp1f1(-0.5*k, 0.5, -0.5*(mu/sigma)**2)


def gamma_moments(n, k, theta, central=False):
    """
    Moments of the Beta-distribution with density

    .. math::

        f(x) = \\frac{x^{k - 1} e^{-x/\\theta}}
                      {\Gamma(k) \\theta^k},

    where :math:`\Gamma` is the gamma function. The central moments are computed by a linear moment transformation as no explicit formulae for the central moments are available (other than recursive formulas, see e.g. Ref. [:cite:label:`Willink_2003`], which are merely linear transforms of the non-central moments).

    Parameters
    ----------
    n : int
        Number of moments.
    k : float
        Shape parameter of the Gamma distribution, must be positive.
    theta : float
        Scale parameter of the Gamma distribution, must be positive.
    central : bool, optional
        Indicates if the central moments shall be computed (default: False).

    Returns
    -------
    mom : array
        Array of raw or central moments up to the (n-1)th order.

    References
    ----------
        +----------------+----------------------+
        | [Willink_2003] | :cite:`Willink_2003` |
        +----------------+----------------------+
        | [Forbes_2011]  | :cite:`Forbes_2011`  |
        +----------------+----------------------+

    """
    # Raw moments about the origin, see Ref. [Forbes_2011], Chapter 22
    j = np.arange(n)
    mom = theta**j * gamma_func(j + k)/gamma_func(k)
    if not central:
        return mom
    # Linear transformation (shift by -mean) to obtain central moments
    mean = k*theta
    return linear_transform(mom, 1, -mean)


def laplace_moments(n, mu, b, central=False):
    """
    Raw or central moments of the Beta-distribution with density

    .. math::

        f(x) = \\frac{1}{2b} e^{-\\abs{x-\mu}/b},

    where :math:`\mu` and :math:`b > 0` are the location and scale parameter, respectively.

    Parameters
    ----------
    mu : float
        Location parameter of the Laplace-distribution, must be real.
    b : float
        Scale parameter of the Laplace-distribution, must be positive.
    central : bool, optional
        Indicates if the central moments shall be computed (default: False).

    Returns
    -------
    mom : array
        Array of raw or central moments up to the (n-1)th order.

    References
    ----------
        +----------------+----------------------+
        | [Forbes_2011]  | :cite:`Forbes_2011`  |
        +----------------+----------------------+

    """
    k = np.arange(n)
    # Central moments from Ref. [Forbes_2011], Chapter 27
    mom = np.zeros(n)
    mom[::2] = factorial(k[::2])*b**k[::2]
    if central:
        return mom
    # For simplicity apply linear transform for moments about the origin
    return linear_transform(mom, 1, -mu)
