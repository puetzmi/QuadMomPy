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
Module for the solution of the one-dimensional Fokker-Planck equation with QBMMs.

"""
import numpy as np

class FokkerPlanckEq1D:
    """
    Class for the numerical solution of the moments equations derived from the one-dimensional Fokker-Planck equation

    .. math::

        f(x) = -\\frac{\\partial}{\\partial \\xi} \\left[ a(\\xi) n(\\xi, t) \\right] - \\frac{\\partial}{\\partial \\xi} \\left[ b(\\xi) n(\\xi, t) \\right] + \\frac{1}{2} \\frac{\\partial^2}{\\partial \\xi^2} \\left[ \\sigma^2(\\xi) n(\\xi, t) \\right],

    where :math:`n` is the number density function of the internal coordinate :math:`\\xi` and time :math:`t`. The function :math:`a` is the drift function, :math:`b` a second drift function due to the Stratonovich interpretation of the underlying Langevin equation (see [:cite:label:`Risken_1989`]), which may be zero, and `sigma` is the noise coefficient in the Langevin equation. For the derived moment equations, see Section 3 in Ref. [:cite:label:`Puetz_2022`].

    Parameters
    ----------
    a : callable
        Drift function.
    b : callable
        Noise-induced drift function.
    sigma : callable
        Noise function.
    qbmm : UnivariateQbmm
        Univariate QBMM, i.e. an object of a subtype of `UnivariateQbmm`, taking a set of univariate moments and returning a quadrature.
    integrate : callable
        Function for temporal integration (see. `Attributes` for additional information).

    Attributes
    ----------
    a : callable
        Drift function.
    b : callable
        Noise-induced drift function.
    sigma : callable
        Noise function.
    qbmm : UnivariateQbmm
        Univariate QBMM, i.e. an object of a subtype of `UnivariateQbmm`, taking a set of univariate moments and returning a quadrature.
    integrate : callable
        Function for temporal integration. It must have the form ``integrate(mom, a, b, sigma, dt, tspan, inv)`` (and possibly further keyword arguments), where `mom` is a set of moments, `a`, `b`, and `sigma` correspond to the variables above, `inv` is a callable that computes a quadrature from a set of univariate moments, `dt` is the numerical step size and `tspan` is the time interval where the equation is to be solved.

    References
    ----------
        +---------------+---------------------+
        | [Risken_1989] | :cite:`Risken_1989` |
        +---------------+---------------------+
        | [Puetz_2022]  | :cite:`Puetz_2022`  |
        +---------------+---------------------+

    """
    def __init__(self, a, b, sigma, qbmm, integrate, **kwargs):
        self.a = a
        self.b = b
        self.sigma = sigma
        self.qbmm = qbmm
        self.integrate = integrate

    def solve(self, mom0, t, dt, adaptive=False, save_all=False, print_time=False, **kwargs):
        """
        Solve the moment equations derived from the one-dimensional Fokker-Planck equation on the interval `[t[0],t[1]]`.

        Parameters
        ----------
        mom0 : array
            Initial moments, i.e. the moment sequence corresponding to `t[0]`.
        t : array_like
            Times at which the Fokker-Planck equation is to be solved. If `adaptive == False`, `t` should be consistent with the step size `dt` such that `(t[k] - t[k-1])/dt` is a positive integer for all `0 < k < len(t)`.
        dt : float
            Time step size. If `adaptive == True` this is only the initial step size.
        adaptive : bool, optional
            Indicates if adaptive step size control is to be used. Default is `False`.
        save_all : bool, optional
            If `save_all == True` the solutions at all computed time steps are saved to the output array (caution with regards to memory is advised in that case). If `save_all == False`, only the solutions that correspond to times in the array `t` are stored in the output. Default is `False`.
        print_time : bool, optional
            Indicates if intermediate values of `t` are to be printed during computation. Default is `False`.
        kwargs :
            Additional parameters passed to the solver.

        Returns
        -------
        mom_all : array
            Solutions of the moment equations at all specified times.
        t_all : array
            All times corresponding to `mom_all`.

        """
        mom_all = [mom0.copy()]
        t_all = [t[0]]

        for n in range(len(t) - 1):
            if print_time:
                print("t = {0:7.6e}".format(t[n+1]))
            tspan = [t[n], t[n+1]]
            m_, t_ = self.integrate(mom_all[-1], self.a, self.b, self.sigma, dt, tspan, self.qbmm, \
                    adaptive=adaptive, **kwargs)
            if save_all:
                mom_all.extend(m_[:])
                t_all.extend(t_[:])
            else:
                mom_all.append(m_[-1])
                t_all.append(t_[-1])

        return np.array(mom_all), np.array(t_all)
