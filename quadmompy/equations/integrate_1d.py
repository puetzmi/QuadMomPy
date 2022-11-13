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
This module contains the RKSSP-methods as well as auxiliary functions to solve the system of moment equations resulting from the one-dimensional Fokker-Planck equation.

"""
import numpy as np
from math import sqrt

def _adapt_stepsize(y0, y1_h, y1_h2, \
        h, p, hmax, atol, rtol, fhmin=0.2, fhmax=2., fhsaf=0.9):
    """
    Check solution with respect to given tolerance and adapt time step accordingly, based on step doubling. The step size is calculated as described in [:cite:label:`Hairer_1993`].

    Parameters
    ----------
    y0 : array
        Solution at previous time step.
    y1_h : array
        Solution at current time step, computed with full step size h.
    y1_h2 : array
        Solution at current time step, computed with half step size h/2.
    h : float
        Current step size.
    p : int
        Order of the used scheme.
    atol : float
        Absolute tolerance.
    rtol : float
        Relative tolerance.
    hmax : float
        Maximum step size.
    fhmin : float
        Minimum factor for calculation of the new step size.
    fhmax : float
        Maximum factor for calculation of the new step size.
    fhsaf : float
        Safety factor for new step size to avoid excessive function evaluations.

    Returns
    -------
    h_new : float
        New step size.
    accept : bool
        Boolean that indicates if step was successful (error within tolerance).

    References
    ----------
        +---------------+---------------------+
        | [Hairer_1993] | :cite:`Hairer_1993` |
        +---------------+---------------------+

    """
    n = len(y0)

    # Estimate error from solutions with step sizes h and h/2, (4.4) in Ref. [Hairer_1993]
    err_est = (y1_h2 - y1_h)/(2**p - 1)

    # Scaling factor based on absolute and relative tolerance, (4.10) in Ref. [Hairer_1993]
    sc = atol + np.maximum(abs(y0), abs(y1_h))*rtol

    # Compute error norm, (4.11) in Ref. [Hairer_1993]
    err = np.linalg.norm(err_est/sc)/n**0.5

    # Check if the error is within the given tolerance and compute new step size
    # (4.13) in Ref. [Hairer_1993]
    if err == 0.:
        hnew = h*fhmax
        return hnew, True
    if err < 1.:
        hnew = h*min(fhmax, max(fhmin, fhsaf*err**(-1/(p+1))))
        return hnew, True
    return h*max(fhmin, fhsaf*err**(-1/(p+1))), False


def _hermite_quad(N):
    """
    Compute N nodes and normalized weights of the Gauss-Hermite quadrature from the recurrence coefficients of the Hermite polynomials as described in Ref. [:cite:label:`Gautschi_2004`].

    Parameters
    ----------
    N : int
        Number of quadrature nodes.

    Returns
    -------
    x : array
        Abscissas of the N-node Gauss-Hermite quadrature.
    w : array
        Normalized weights of the N-node Gauss-Hermite quadrature.

    References
    ----------
        +-----------------+-----------------------+
        | [Gautschi_2004] | :cite:`Gautschi_2004` |
        +-----------------+-----------------------+

    """
    # Beta recurrence coefficients 1..N-1
    # (beta[0] is not needed; the alpha coefficients are all zero, see Ref. [Gautschi_2004])
    beta = 0.5*np.arange(N, dtype=float)

    # Assemble Jacobi matrix
    J = np.zeros((N,N))
    for i in range(1,N):
        J[i,i-1] = sqrt(beta[i])
        #J[i-1,i] = sqrt(beta[i])   # numpy.eigh only needs lower triangular matrix

    # Compute eigenvalues (abscissas of the quadrature) and eigenvalues
    x, eigvecs = np.linalg.eigh(J)

    # Compute weights from eigenvectors (already normalized)
    w = eigvecs[0]**2

    return x, w


def _normal_moments(nmom, x, w, mu, sigma):
    """
    Compute the first `nmom` moments of a normal distribution given the nodes and normalized weights of a Gauss-Hermite quadrature. For an exact calculation, `nom/2` nodes and weights must be provided, assuming even `nmom`.

    Parameters
    ----------
    nmom : int
        Number of moments
    x : array
        Nodes of a Hermite quadrature.
    w : array
        Normalized weights of a Hermite quadrature.
    mu : float
        Mu parameter of the normal distribution (corresponds to the mean).
    sigma : float
        Sigma parameter of the normal distribution (corresponds to the standard deviation).

    Returns
    -------
    mom : array
        The first `nmom` moments of a normal distribution with parameters `mu` and `sigma`.

    """
    # Transform nodes from standard Hermite weight function to normal PDF with parameters mu and sigma
    xt = sqrt(2)*sigma*x + mu

    # Assemble Vandermonde matrix containing powers of the nodes
    V = np.vander(xt, nmom, increasing=True).T

    # Compute moments from Vandermonde matrix and weights
    mom = V@w
    return mom


def rk2ssp(mom, a, b, sigma, dt, tspan, inv, adaptive=False, dtmax=1e100, atol=1e-7, rtol=1e-5, fhmin=0.2, fhmax=2., fhsaf=0.9):
    """
    Use RK2SSP scheme, see Ref. [:cite:label:`Shu_1988`], to compute solution to one-dimensional moment equations derived from a Fokker-Planck equation with drift and noise-induced advection and diffusion terms. If the adaptive version is selected, the step size is adjusted using an error estimate based on step douling and Richardson extrapolation, see Ref. [:cite:label:`Hairer_1993`].

    Parameters
    ----------
    mom : array
        Valid sequence of initial moments.
    a : callable
        Drift function.
    b : callable
        Noise-induced drift function.
    sigma : callable
        Noise function.
    dt : float
        Time step size (may change if 'adaptive' is true.
    tspan : array
        Start and end times.
    inv : callable
        Inversion algorithm taking a set of univariate moments and returning a quadrature.
    adaptive : bool, optional
        Specifies if adaptive step size control is used, false by default.
    dtmax : float, optional
        Maximum step size
    atol : float, optional
        Absolute tolerance (only for adaptive step size control).
    rtol : float, optional
        Relative tolerance (only for adaptive step size control).
    fhmin : float, optional
        Minimum factor for calculation of the new step size (only for adaptive step size control).
    fhmax : float, optional
        Maximum factor for calculation of the new step size (only for adaptive step size control).
    fhsaf : float, optional
        Safety factor for new step size to avoid excessive function evaluations (only for adaptive step size control).

    Returns
    -------
    mom_all : List
        Moment sequences for all time steps.
    t_all : List
        Time values for all time steps.

    References
    ----------
        +---------------+---------------------+
        | [Shu_1988]    | :cite:`Shu_1988`    |
        +---------------+---------------------+
        | [Hairer_1993] | :cite:`Hairer_1993` |
        +---------------+---------------------+

    """
    m = mom
    t = tspan[0]
    _2N = len(m)

    # Initialize lists containing all times and moment sets
    t_all = []
    mom_all = []

    # Initialize arrays for advection and diffusion terms
    mdot_adv_drift = np.zeros(_2N)
    mdot_adv_noise = np.zeros(_2N)
    mdot_diff = np.zeros(_2N)

    # Start time integration
    while t <= tspan[-1]:
        success = False

        while not success:
            # RK2SSP-step with step size dt
            m_ = m.copy()
            # Compute intermediate moments
            for _ in range(2):
                # Moment inversion
                v, w = inv(m_)
                v[np.isclose(v, 0)] = 0.
                # Compute advection and diffusion terms
                mdot_adv_drift[1:] = [k*a(v)*v**(k-1)@w for k in range(1,_2N)]
                mdot_adv_noise[1:] = [k*b(v)*v**(k-1)@w for k in range(1,_2N)]
                mdot_diff[2:] = [0.5*k*(k-1)*sigma(v)**2*v**(k-2)@w for k in range(2,_2N)]
                # Update moments
                m_ += dt*(mdot_adv_drift + mdot_adv_noise + mdot_diff)
            # Compute average
            m_dt = 0.5*(m + m_)

            # Finish step if step size is not adaptive
            if not adaptive: break

            # Now compute two steps with step size dt/2
            # (For the sake of simplicity, the drift and advection terms at time t are computed again)
            dt2 = 0.5*dt
            m_dt2 = m.copy()
            for _ in range(2):
                # RK2SSP-step with step size dt/2
                m_ = m_dt2.copy()
                # Compute intermediate moments
                for _ in range(2):
                    # Moment inversion
                    v, w = inv(m_)
                    v[np.isclose(v, 0)] = 0.
                    # Compute advection and diffusion terms
                    mdot_adv_drift[1:] = [k*a(v)*v**(k-1)@w for k in range(1,_2N)]
                    mdot_adv_noise[1:] = [k*b(v)*v**(k-1)@w for k in range(1,_2N)]
                    mdot_diff[2:] = [0.5*k*(k-1)*sigma(v)**2*v**(k-2)@w for k in range(2,_2N)]
                    # Update moments
                    m_ += dt2*(mdot_adv_drift + mdot_adv_noise + mdot_diff)
                # Compute average
                m_dt2 = 0.5*(m_dt2 + m_)

            # Check if solution is within tolerance and compute new step size as described in Ref. [Hairer_1993]
            dt, success = _adapt_stepsize(m, m_dt, m_dt2, dt, 2, dtmax, atol, rtol, fhmin, fhmax, fhsaf)

        # Assign updated solution
        m = m_dt.copy()

        # Progress in time
        if t == tspan[-1]: break
        dt = min(dt, dtmax, tspan[-1] - t)
        t += dt

        # Append current solution and time
        mom_all.append(m.copy())
        t_all.append(t)

    return mom_all, t_all


def rk2ssp(mom, a, b, sigma, dt, tspan, inv, adaptive=False, dtmax=1e100, atol=1e-7, rtol=1e-5, fhmin=0.2, fhmax=2., fhsaf=0.9):
    """
    Use RK2SSP scheme, see Ref. [:cite:label:`Shu_1988`], to compute solution to one-dimensional moment equations derived from a Fokker-Planck equation with drift and noise-induced advection and diffusion terms. If the adaptive version is selected, the step size is adjusted using an error estimate based on step douling and Richardson extrapolation, see Ref. [:cite:label:`Hairer_1993`].

    Parameters
    ----------
    mom : array
        Valid sequence of initial moments.
    a : callable
        Drift function.
    b : callable
        Noise-induced drift function.
    sigma : callable
        Noise function.
    dt : float
        Time step size (may change if 'adaptive' is true.
    tspan : array
        Start and end times.
    inv : callable
        Inversion algorithm taking a set of univariate moments and returning a quadrature.
    adaptive : bool, optional
        Specifies if adaptive step size control is used, false by default.
    dtmax : float, optional
        Maximum step size
    atol : float, optional
        Absolute tolerance (only for adaptive step size control).
    rtol : float, optional
        Relative tolerance (only for adaptive step size control).
    fhmin : float, optional
        Minimum factor for calculation of the new step size (only for adaptive step size control).
    fhmax : float, optional
        Maximum factor for calculation of the new step size (only for adaptive step size control).
    fhsaf : float, optional
        Safety factor for new step size to avoid excessive function evaluations (only for adaptive step size control).

    Returns
    -------
    mom_all : List
        Moment sequences for all time steps.
    t_all : List
        Time values for all time steps.

    References
    ----------
        +---------------+---------------------+
        | [Shu_1988]    | :cite:`Shu_1988`    |
        +---------------+---------------------+
        | [Hairer_1993] | :cite:`Hairer_1993` |
        +---------------+---------------------+

    """
    m = mom
    t = tspan[0]
    _2N = len(m)

    # Initialize lists containing all times and moment sets
    t_all = []
    mom_all = []

    # Initialize arrays for advection and diffusion terms
    mdot_adv_drift = np.zeros(_2N)
    mdot_adv_noise = np.zeros(_2N)
    mdot_diff = np.zeros(_2N)

    # Start time integration
    while t <= tspan[-1]:
        success = False

        while not success:
            # RK2SSP-step with step size dt
            m_ = m.copy()
            # Compute intermediate moments
            for _ in range(2):
                # Moment inversion
                v, w = inv(m_)
                v[np.isclose(v, 0)] = 0.
                # Compute advection and diffusion terms
                mdot_adv_drift[1:] = [k*a(v)*v**(k-1)@w for k in range(1,_2N)]
                mdot_adv_noise[1:] = [k*b(v)*v**(k-1)@w for k in range(1,_2N)]
                mdot_diff[2:] = [0.5*k*(k-1)*sigma(v)**2*v**(k-2)@w for k in range(2,_2N)]
                # Update moments
                m_ += dt*(mdot_adv_drift + mdot_adv_noise + mdot_diff)
            # Compute average
            m_dt = 0.5*(m + m_)

            # Finish step if step size is not adaptive
            if not adaptive: break

            # Now compute two steps with step size dt/2
            # (For the sake of simplicity, the drift and advection terms at time t are computed again)
            dt2 = 0.5*dt
            m_dt2 = m.copy()
            for _ in range(2):
                # RK2SSP-step with step size dt/2
                m_ = m_dt2.copy()
                # Compute intermediate moments
                for _ in range(2):
                    # Moment inversion
                    v, w = inv(m_)
                    v[np.isclose(v, 0)] = 0.
                    # Compute advection and diffusion terms
                    mdot_adv_drift[1:] = [k*a(v)*v**(k-1)@w for k in range(1,_2N)]
                    mdot_adv_noise[1:] = [k*b(v)*v**(k-1)@w for k in range(1,_2N)]
                    mdot_diff[2:] = [0.5*k*(k-1)*sigma(v)**2*v**(k-2)@w for k in range(2,_2N)]
                    # Update moments
                    m_ += dt2*(mdot_adv_drift + mdot_adv_noise + mdot_diff)
                # Compute average
                m_dt2 = 0.5*(m_dt2 + m_)

            # Check if solution is within tolerance and compute new step size as described in Ref. [Hairer_1993]
            dt, success = _adapt_stepsize(m, m_dt, m_dt2, dt, 2, dtmax, atol, rtol, fhmin, fhmax, fhsaf)

        # Assign updated solution
        m = m_dt.copy()

        # Progress in time
        if t == tspan[-1]: break
        dt = min(dt, dtmax, tspan[-1] - t)
        t += dt

        # Append current solution and time
        mom_all.append(m.copy())
        t_all.append(t)

    return mom_all, t_all


def rk2ssp_ar(mom, a, b, sigma, dt, tspan, inv, adaptive=False, dtmax=1e100, atol=1e-7, rtol=1e-5, fhmin=0.2, fhmax=2., fhsaf=0.9):
    """
    Use absolutely realizable RK2SSP scheme, see [:cite:label:`Puetz_2022`], a realizability-preserving modification of the normal RK2SSP (see method :meth:`rk2ssp`) to compute solution to one-dimensional moment equations derived from a Fokker-Planck equation with drift and noise-induced advection and diffusion terms.

    Parameters
    ----------
    mom : array
        Valid sequence of initial moments.
    a : callable
        Drift function.
    b : callable
        Noise-induced drift function.
    sigma : callable
        Noise function.
    dt : float
        Time step size (may change if 'adaptive' is true.
    tspan : array
        Start and end times.
    inv : callable
        Inversion algorithm taking a set of univariate moments and returning a quadrature.
    adaptive : bool, optional
        Specifies if adaptive step size control is used, false by default.
    dtmax : float, optional
        Maximum step size
    atol : float, optional
        Absolute tolerance (only for adaptive step size control).
    rtol : float, optional
        Relative tolerance (only for adaptive step size control).
    fhmin : float, optional
        Minimum factor for calculation of the new step size (only for adaptive step size control).
    fhmax : float, optional
        Maximum factor for calculation of the new step size (only for adaptive step size control).
    fhsaf : float, optional
        Safety factor for new step size to avoid excessive function evaluations (only for adaptive step size control).

    Returns
    -------
    mom_all : List
        Moment sequences for all time steps.
    t_all : List
        Time values for all time steps.

    References
    ----------
        +---------------+---------------------+
        | [Puetz_2022]  | :cite:`Puetz_2022`  |
        +---------------+---------------------+
        +---------------+---------------------+
        | [Hairer_1993] | :cite:`Hairer_1993` |
        +---------------+---------------------+

    """
    m = mom
    t = tspan[0]
    _2N = len(m)
    N = _2N//2

    # Initialize lists containing all times and moment sets
    t_all = []
    mom_all = []

    # Compute Hermite quadrature for moments of normal distribution in advance
    xh, wh = _hermite_quad(N)

    # Start time integration
    while t <= tspan[-1]:
        dt = min(tspan[-1] - t, dt)
        success = False

        # Adapt step size if needed
        while not success:
            # RK2SSP-step with step size dt
            m_ = m.copy()
            # Compute intermediate moments
            for _ in range(2):
                # Moment inversion
                v, w = inv(m_)
                v[np.isclose(v, 0)] = 0.
                v0 = v.copy()
                # Apply drift terms to quadrature nodes
                v += (a(v) + b(v))*dt
                # Update moments considering diffusion (normal moments)
                norm_moments = np.array([ \
                        _normal_moments(_2N, xh, wh, mu=v[i], sigma=sigma(v0[i])*sqrt(dt)) \
                        for i in range(len(v))]).T
                m_ = norm_moments@w
            # Compute average
            m_dt = 0.5*(m + m_)

            # Finish step if step size is not adaptive
            if not adaptive: break

            # Now compute two steps with step size dt/2
            # (For the sake of simplicity, the drift and advection terms at time t are computed again)
            dt2 = 0.5*dt
            m_dt2 = m.copy()
            for _ in range(2):
                # RK2SSP-step with step size dt/2
                m_ = m_dt2.copy()
                # Compute intermediate moments
                for _ in range(2):
                    # Moment inversion
                    v, w = inv(m_)
                    v[np.isclose(v, 0)] = 0.
                    v0 = v.copy()
                    # Apply drift terms to quadrature nodes
                    v += (a(v) + b(v))*dt2
                    # Update moments considering diffusion (normal moments)
                    norm_moments = np.array([ \
                            _normal_moments(_2N, xh, wh, mu=v[i], sigma=sigma(v0[i])*sqrt(dt2)) \
                            for i in range(len(v))]).T
                    m_ = norm_moments@w
                # Compute average
                m_dt2 = 0.5*(m_dt2 + m_)

            # Check if solution is within tolerance and compute new step size as described in Ref. [Hairer_1993]
            dt, success = _adapt_stepsize(m, m_dt, m_dt2, dt, 2, dtmax, atol, rtol, fhmin, fhmax, fhsaf)

        # Assign updated solution
        m = m_dt.copy()

        # Progress in time
        if t == tspan[-1]: break
        dt = min(dt, dtmax, tspan[-1] - t)
        t += dt

        # Append current solution and time
        mom_all.append(m.copy())
        t_all.append(t)

    return mom_all, t_all
