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
This module contains the improved EQMOM root finding algorithms based on moment realizbility proposed by Pigou et al. [:cite:label:`Pigou_2018`].

"""
import numpy as np
from quadmompy.moments.transform import rc2canonmom, rc2zeta, zeta2rc

def pigou_hamburger(sigma0, sigma_range, mom, m2ms, ms2m, inv, atol, rtol, full_output=False, maxiter=100):
    """
    The moment-realizability-based root finding algorithm for a Hamburger problem proposed by Pigou et al. [:cite:label:`Pigou_2018`] (algorithm in Section 3.3).

    Parameters
    ----------
    sigma0 : float
        Initial guess for sigma.
    sigma_range : tuple
        Interval of valid sigma values.
    mom : (2*N + 1,) array
        Set of 2*N + 1 realizable moments, where N is the number of KDFs in EQMOM.
    m2ms : callable
        Function that converts ordinary moments m to degenerated moments m*, for details see Refs. [:cite:label:`Pigou_2018`] and [:cite:label:`Yuan_2012`]. It must be a function of the form ``f(m, sigma)``, where `m` are the ordinary moments and `sigma` is the current EQMOM-parameter (float).
    ms2m : callable
        Function that converts degenerated moments m* to ordinary moments m, for details see Refs. [:cite:label:`Pigou_2018`] and [:cite:label:`Yuan_2012`]. It must be a function of the form ``f(ms, sigma)``, where `ms` are the degenerated moments and `sigma` is the current EQMOM-parameter (float).
    inv : MomentInversion
        Object of subtype of `MomentInversion`.
    atol : float
        Allowed absolute error determining termination of the root search.
    rtol : float
        Allowed relative error determining termination of the root search.
    full_output : bool, optional
        If true, the function additionally returns a dictionary containing intermediate values.
    maxiter : int, optional
        Maximum number of iterations. When exceeded, the algorithm is terminated with a `RuntimeError` (default: 100).

    Returns
    -------
    sigma : float
        The computed root satisfying the given conditions.
    xi_first : array
        Nodes of the first quadrature
    w_first : array
        Weights of the first quadrature
    results : dict, optional
        Dictionary with additional details such as intermediate values and convergence information (returned if full_output is true).

    Raises
    ------
    RuntimeError
        If moments are not realizable or if the algorithm has not converged after `maxiter` iterations.

    References
    ----------
        +--------------+--------------------+
        | [Pigou_2018] | :cite:`Pigou_2018` |
        +--------------+--------------------+
        | [Yuan_2012]  | :cite:`Yuan_2012`  |
        +--------------+--------------------+

    """
    rc_func = inv.recurrence_coeffs
    m = mom

    # Initialize different sigmas with subscripts as described in Ref. [Pigou_2018], Section 3.3
    keys = ['l', 't1', 't2', 'r']
    sigma = {key: {0: 0.} for key in keys}

    # Vector b*(sigma) (recurrence coefficients) for each of the sigmas above
    bs = {key: None for key in keys}

    # Check if the given moment sequence is on the moment space boundary. If so, sigma is 0.
    a0, b0 = rc_func(m)
    if np.any(abs(b0) < atol):
        b0[abs(b0) < atol] = 0.
        sigma = 0.
        xi, w = inv.quad_from_rc(a0, b0)
        if full_output:
            return sigma, xi, w, {'nit': 0}
        return sigma, xi, w


    # The numbers below refer to those in the algorithm in Ref. [Pigou_2018], Section 3.3
    # (1) Check realizability of the moment set
    realizable = np.all(b0 >= 0.)
    if not realizable:
        msg = "Moment set is not realizable.\n"
        msg += "m = {0!s}\n".format(m)
        msg += "b = {0!s}".format(b0)
        raise RuntimeError(msg)

    # (2) Initialize interval
    # (the analytical solution for 3 moments should be provided as parameter `sigma_range[1]`)
    sigma['l'][0], sigma['r'][0] = sigma_range

    # (3) Iterate over k
    success = False
    for k in range(1, maxiter + 1):

        # (3a) Identify the index of the first negative element of b*(sigma['r'])
        _, bs['r'] = rc_func(m2ms(m, sigma['r'][k-1]))
        try:
            j = np.where(bs['r'] < 0)[0][0]
        except IndexError:
            if abs(bs['r'][-1]) < atol:
                sigma['t1'][k] = sigma['l'][k-1]
                sigma['l'][k] = sigma['l'][k-1]
                sigma['r'][k] = sigma['r'][k-1]
                success = True
                break
            return pigou_hamburger(sigma0, sigma_range, m[:-2], m2ms, ms2m, rc_func, atol, rtol, full_output, maxiter)

        # (3b) Compute sigma[t1] and b*(sigma[t1])
        sigma['t1'][k] = 0.5*(sigma['l'][k-1] + sigma['r'][k-1])
        _, bs['t1'] = rc_func(m2ms(m, sigma['t1'][k]))

        # (3c) Compute sigma[t2] and b*(sigma[t2])
        _, bs['l'] = rc_func(m2ms(m, sigma['l'][k-1]))
        sigma['t2'][k] = sigma['t1'][k] + (sigma['t1'][k] - sigma['l'][k-1]) \
                        * bs['t1'][j] / (bs['t1'][j]**2 - bs['l'][j]*bs['r'][j])**0.5
        _, bs['t2'] = rc_func(m2ms(m, sigma['t2'][k]))

        # (3d) Set lower boundary to max(sigma[l], sigma[t1], sigma[t2]) such that b*
        # contains only positive values.
        sigma['l'][k] = sigma['l'][k-1]
        for key in ['t1', 't2']:
            if np.all(bs[key][1:] > 0):
                sigma['l'][k] = max(sigma['l'][k], sigma[key][k])

        # (3e) Set upper boundary to min(sigma[t1], sigma[t2], sigma[r]) such that b*
        # contains at least one negative value.
        # (considering sigma[r] = max(sigma))
        sigma['r'][k] = sigma['r'][k-1]
        for key in ['t1', 't2']:
            if np.any(bs[key][1:] < 0):
                sigma['r'][k] = min(sigma['r'][k], sigma[key][k])

        # Compute error assuming the worst case and terminate procedure if criteria are met
        eabs_max = sigma['r'][k] - sigma['l'][k]
        success = eabs_max < (atol + rtol*sigma['l'][k])
        if success: break

    # Raise error if algorithm has failed to converge after `maxiter` iterations
    if not success:
        err_msg = "Maximum number of iterations ({0:d}) exceeded.\n".format(maxiter)
        err_msg += "Current sigma interval: ({0:8.7e}, {0:8.7e})".format(sigma['l'][k-1], sigma['r'][k-1])
        raise RuntimeError(err_msg)

    nit = k
    xi, w = inv(m2ms(m[:-1], sigma['l'][k]))
    if full_output:
        results = {'sigma_{0:s}'.format(key): np.array([sigma[key][i] for i in range(nit)]) for key in ['l','r']}
        results['nit'] = nit
        return sigma['l'][k], xi, w

    return sigma['l'][k], xi, w


def pigou_stieltjes(sigma0, sigma_range, mom, m2ms, ms2m, inv, atol, rtol, full_output=False, maxiter=100):
    """
    The moment-realizability-based root finding algorithm for a Stieltjes problem proposed by Pigou et al. [:cite:label:`Pigou_2018`] (algorithm in Section 3.4).

    Parameters
    ----------
    sigma0 : float
        Initial guess for sigma.
    sigma_range : tuple
        Interval of valid sigma values.
    mom : (2*N + 1,) array
        Set of 2*N + 1 realizable moments, where N is the number of KDFs in EQMOM.
    m2ms : callable
        Function that converts ordinary moments m to degenerated moments m*, for details see Refs. [:cite:label:`Pigou_2018`] and [:cite:label:`Yuan_2012`]. It must be a function of the form ``f(m, sigma)``, where `m` are the ordinary moments and `sigma` is the current EQMOM-parameter (float).
    ms2m : callable
        Function that converts degenerated moments m* to ordinary moments m, for details see Refs. [:cite:label:`Pigou_2018`] and [:cite:label:`Yuan_2012`]. It must be a function of the form ``f(ms, sigma)``, where `ms` are the degenerated moments and `sigma` is the current EQMOM-parameter (float).
    inv : MomentInversion
        Object of subtype of `MomentInversion`.
    atol : float
        Allowed absolute error determining termination of the root search.
    rtol : float
        Allowed relative error determining termination of the root search.
    full_output : bool, optional
        If true, the function additionally returns a dictionary containing intermediate values.
    maxiter : int, optional
        Maximum number of iterations. When exceeded, the algorithm is terminated with a `RuntimeError` (default: 100).

    Returns
    -------
    sigma : float
        The computed root satisfying the given conditions.
    xi_first : array
        Nodes of the first quadrature
    w_first : array
        Weights of the first quadrature
    results : dict, optional
        Dictionary with additional details such as intermediate values and convergence information (returned if full_output is true).

    Raises
    ------
    RuntimeError
        If moments are not realizable or if the algorithm has not converged after `maxiter` iterations.

    References
    ----------
        +--------------+--------------------+
        | [Pigou_2018] | :cite:`Pigou_2018` |
        +--------------+--------------------+
        | [Yuan_2012]  | :cite:`Yuan_2012`  |
        +--------------+--------------------+

    """
    rc_func = inv.recurrence_coeffs
    m = mom

    # Initialize different sigmas with subscripts as described in Ref. [Pigou_2018], Section 3.4
    keys = ['l', 't1', 't2', 'r']
    sigma = {key: {0: 0.} for key in keys}

    # Vector zeta*(sigma) (continued-fraction coefficients) for each of the sigmas above
    zetas = {key: None for key in keys}

    # Check if the given moment sequence is on the moment space boundary. If so, sigma is 0.
    zeta0 = rc2zeta(*rc_func(m))
    if np.any(abs(zeta0) < atol):
        zeta0[abs(zeta0) < atol] = 0.
        sigma = 0.
        xi, w = inv.quad_from_rc(*zeta2rc(zeta0))
        if full_output:
            return sigma, xi, w, {'nit': 0}
        return sigma, xi, w


    # The numbers below refer to those in the algorithm in Ref. [Pigou_2018], Section 3.4
    # (1) Check realizability of the moment set
    realizable = np.all(zeta0 >= 0.)
    if not realizable:
        msg = "Moment set is not realizable.\n"
        msg += "m = {0!s}\n".format(m)
        msg += "zeta = {0!s}".format(zeta0)
        raise RuntimeError(msg)

    # (2) Initialize interval
    # (the analytical solution for 3 moments should be provided as parameter `sigma_range[1]`)
    sigma['l'][0], sigma['r'][0] = sigma_range

    # (3) Iterate over k
    success = False
    for k in range(1, maxiter + 1):

        # (3a) Identify the index of the first negative element of zeta*(sigma['r'])
        zetas['r'] = rc2zeta(*rc_func(m2ms(m, sigma['r'][k-1])))
        try:
            j = np.where(zetas['r'] < 0)[0][0]
        except IndexError:
            if abs(zetas['r'][-1]) < atol:
                sigma['t1'][k] = sigma['l'][k-1]
                sigma['l'][k] = sigma['l'][k-1]
                sigma['r'][k] = sigma['r'][k-1]
                success = True
                break
            return pigou_stieltjes(sigma0, sigma_range, m[:-2], m2ms, ms2m, inv, atol, rtol, full_output, maxiter)

        # (3b) Compute sigma[t1] and zeta*(sigma[t1])
        sigma['t1'][k] = 0.5*(sigma['l'][k-1] + sigma['r'][k-1])
        zetas['t1'] = rc2zeta(*rc_func(m2ms(m, sigma['t1'][k])))

        # (3c) Compute sigma[t2] and zeta*(sigma[t2])
        zetas['l'] = rc2zeta(*rc_func(m2ms(m, sigma['l'][k-1])))
        sigma['t2'][k] = sigma['t1'][k] + (sigma['t1'][k] - sigma['l'][k-1]) \
                        * zetas['t1'][j] / (zetas['t1'][j]**2 - zetas['l'][j]*zetas['r'][j])**0.5
        zetas['t2'] = rc2zeta(*rc_func(m2ms(m, sigma['t2'][k])))

        # (3d) Set lower boundary to max(sigma[l], sigma[t1], sigma[t2]) such that zetas*
        # contains only positive values.
        sigma['l'][k] = sigma['l'][k-1]
        for key in ['t1', 't2']:
            if np.all(zetas[key] > 0):
                sigma['l'][k] = max(sigma['l'][k], sigma[key][k])

        # (3e) Set upper boundary to min(sigma[t1], sigma[t2], sigma[r]) such that zetas*
        # contains at least one negative value.
        # (considering sigma[r] = max(sigma))
        sigma['r'][k] = sigma['r'][k-1]
        for key in ['t1', 't2']:
            if np.any(zetas[key][1:] < 0):
                sigma['r'][k] = min(sigma['r'][k], sigma[key][k])

        # Compute error assuming the worst case and terminate procedure if criteria are met
        eabs_max = sigma['r'][k] - sigma['l'][k]
        success = eabs_max < (atol + rtol*sigma['l'][k])
        if success: break

    # Raise error if algorithm has failed to converge after `maxiter` iterations
    if not success:
        err_msg = "Maximum number of iterations ({0:d}) exceeded.\n".format(maxiter)
        err_msg += "Current sigma interval: ({0:8.7e}, {0:8.7e})".format(sigma['l'][k-1], sigma['r'][k-1])
        raise RuntimeError(err_msg)

    nit = k
    xi, w = inv(m2ms(m[:-1], sigma['l'][k]))
    if full_output:
        results = {'sigma_{0:s}'.format(key): np.array([sigma[key][i] for i in range(nit)]) for key in ['l','r']}
        results['nit'] = nit
        return sigma['l'][k], xi, w

    return sigma['l'][k], xi, w


# TODO: Make this more robust
def pigou_hausdorff(sigma0, sigma_range, mom, m2ms, ms2m, inv, atol, rtol, full_output=False, maxiter=100):
    """
    The moment-realizability-based root finding algorithm for a Hausdorff problem proposed by Pigou et al. [:cite:label:`Pigou_2018`] (algorithm in Section 3.5).

    Parameters
    ----------
    sigma0 : float
        Initial guess for sigma.
    sigma_range : tuple
        Interval of valid sigma values.
    mom : (2*N + 1,) array
        Set of 2*N + 1 realizable moments, where N is the number of KDFs in EQMOM.
    m2ms : callable
        Function that converts ordinary moments m to degenerated moments m*, for details see Refs. [:cite:label:`Pigou_2018`] and [:cite:label:`Yuan_2012`]. It must be a function of the form ``f(m, sigma)``, where `m` are the ordinary moments and `sigma` is the current EQMOM-parameter (float).
    ms2m : callable
        Function that converts degenerated moments m* to ordinary moments m, for details see Refs. [:cite:label:`Pigou_2018`] and [:cite:label:`Yuan_2012`]. It must be a function of the form ``f(ms, sigma)``, where `ms` are the degenerated moments and `sigma` is the current EQMOM-parameter (float).
    inv : MomentInversion
        Object of subtype of `MomentInversion`.
    atol : float
        Allowed absolute error determining termination of the root search.
    rtol : float
        Allowed relative error determining termination of the root search.
    full_output : bool, optional
        If true, the function additionally returns a dictionary containing intermediate values.
    maxiter : int, optional
        Maximum number of iterations. When exceeded, the algorithm is terminated with a `RuntimeError` (default: 100).

    Returns
    -------
    sigma : float
        The computed root satisfying the given conditions.
    xi_first : array
        Nodes of the first quadrature
    w_first : array
        Weights of the first quadrature
    results : dict, optional
        Dictionary with additional details such as intermediate values and convergence information (returned if full_output is true).

    Raises
    ------
    RuntimeError
        If moments are not realizable or if the algorithm has not converged after `maxiter` iterations.

    References
    ----------
        +--------------+--------------------+
        | [Pigou_2018] | :cite:`Pigou_2018` |
        +--------------+--------------------+
        | [Yuan_2012]  | :cite:`Yuan_2012`  |
        +--------------+--------------------+

    """
    rc_func = inv.recurrence_coeffs
    m = mom
    N = len(mom) - 1

    # Initialize different sigmas with subscripts as described in Ref. [Pigou_2018], Section 3.5
    keys = ['l', 't1', 't2', 'r']
    sigma = {key: {0: 0.} for key in keys}

    # Vector p*(sigma) (canonical moments) for each of the sigmas above
    ps = {key: None for key in keys}

    # Check if the given moment sequence is on the moment space boundary. If so, sigma is 0.
    p0 = rc2canonmom(*rc_func(m))
    boundary = (abs(p0) < atol) | (abs(p0 - 1) < rtol)
    if np.any(boundary[1:]):                            # TODO this should be tested
        sigma = 0.
        j = np.where(boundary)[0][0]
        xi, w = inv(m[:j])
        return sigma, xi, w
        if full_output:
            return sigma, xi, w, {'nit': 0}
        return sigma, xi, w

    # The numbers below refer to those in the algorithm in Ref. [Pigou_2018], Section 3.5
    # (1) Check realizability of the moment set
    realizable = np.all(p0 <= 1) and np.all(p0 >= 0)
    if not realizable:
        msg = "Moment set is not realizable.\n"
        msg += "m = {0!s}\n".format(m)
        msg += "b = {0!s}".format(b0)
        raise RuntimeError(msg)

    # (2) Initialize interval
    # (the analytical solution for 3 moments should be provided as parameter `sigma_range[1]`)
    sigma['l'][0], sigma['r'][0] = sigma_range

    # (3) Iterate over k
    success = False
    for k in range(1, maxiter + 1):

        # (3a) Identify the index of the first negative element of p*(sigma['r'])
        ps['r'] = rc2canonmom(*rc_func(m2ms(m, sigma['r'][k-1])))

        if (abs(ps['r'][-1]) < atol) or (ps['r'][-1] - rtol > 1):
            sigma['t1'][k] = sigma['l'][k-1]
            sigma['l'][k] = sigma['l'][k-1]
            sigma['r'][k] = sigma['r'][k-1]
            success = True

        # Check if p*[N] is approximately zero
        if abs(ps['r'][-1]) < atol:
            sigma['l'][k] = sigma['r'][k-1]
            success = True
            break

        try:
            j = np.where((ps['r'] < 0) | (ps['r']  > 1))[0][0]
        except IndexError:
            if (abs(ps['r'][-1]) < atol) or (ps['r'][-1] - rtol > 1):
                sigma['t1'][k] = sigma['l'][k-1]
                sigma['l'][k] = sigma['l'][k-1]
                sigma['r'][k] = sigma['r'][k-1]
                success = True
                break
            return pigou_hausdorff(sigma0, sigma_range, m[:-2], m2ms, ms2m, inv, atol, rtol, full_output, maxiter)

        # (3b) Compute sigma[t1] and p*(sigma[t1])
        sigma['t1'][k] = 0.5*(sigma['l'][k-1] + sigma['r'][k-1])
        ps['t1'] = rc2canonmom(*rc_func(m2ms(m, sigma['t1'][k])))

        # (3c)/(3d) Compute sigma[t2] and p*(sigma[t2])
        ps['l'] = rc2canonmom(*rc_func(m2ms(m, sigma['l'][k-1])))
        if j < N and ps['r'][j] > 1:
            qs_j = lambda key: 1 - ps[key][j]
            sigma['t2'][k] = sigma['t1'][k] + (sigma['t1'][k] - sigma['l'][k-1]) \
                        * qs_j('t1')/(qs_j('t1')**2 - qs_j('l')*qs_j('r'))**0.5
        else:
            sigma['t2'][k] = sigma['t1'][k] + (sigma['t1'][k] - sigma['l'][k-1]) \
                        * ps['t1'][j]/(ps['t1'][j]**2 - ps['l'][j]*ps['r'][j])**0.5
        ps['t2'] = rc2canonmom(*rc_func(m2ms(m, sigma['t2'][k])))

        # (3d) Set lower boundary to max(sigma[l], sigma[t1], sigma[t2]) such that p*
        # contains only values in (0,1)
        # (considering sigma[l] = min(sigma))
        sigma['l'][k] = sigma['l'][k-1]
        if np.all(ps['t1'] > 0) and np.all(ps['t1'] < 1):
            sigma['l'][k] = max(sigma['l'][k], sigma['t1'])
        if np.all(ps['t2'][1:] > 0) and np.all(ps['t2'] < 1):
            sigma['l'][k] = max(sigma['l'][k], sigma['t2'])

        # (3e) Set upper boundary to min(sigma[t1], sigma[t2], sigma[r]) such that p*
        # contains at least one value outside (0,1)
        # (considering sigma[r] = max(sigma))
        sigma['r'][k] = sigma['r'][k-1]
        if np.any(ps['t1'] < 0) or np.any(ps['t1'] > 1):
            sigma['r'][k] = min(sigma['r'][k], sigma['t1'][k])
        if np.any(ps['t2'] < 0) or np.any(ps['t2'] > 1):
            sigma['r'][k] = min(sigma['r'][k], sigma['t2'][k])

        # Compute error assuming the worst case and terminate procedure if criteria are met
        eabs_max = sigma['r'][k] - sigma['l'][k]
        success = eabs_max < (atol + rtol*sigma['l'][k])
        if success: break

    # Raise error if algorithm has failed to converge after `maxiter` iterations
    if not success:
        err_msg = "Maximum number of iterations ({0:d}) exceeded.\n".format(maxiter)
        err_msg += "Current sigma interval: ({0:8.7e}, {0:8.7e})".format(sigma['l'][k-1], sigma['r'][k-1])
        raise RuntimeError(err_msg)

    nit = k
    xi, w = inv(m2ms(m[:-1], sigma['l'][k]))
    if full_output:
        results = {'sigma_{0:s}'.format(key): np.array([sigma[key][i] for i in range(nit)]) for key in ['l','r']}
        results['nit'] = nit
        return sigma['l'][k], xi, w

    return sigma['l'][k], xi, w
