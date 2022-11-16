# pylint: disable=import-outside-toplevel,too-many-locals
"""
This module contains test routines that are identical for different tests,

"""
import pytest

def _test_std(nmom, inv_type, inv_setup, x_range):
    """
    Test standard inversion, i.e. the computation of a Gauss quadrature or the
    corresponding recurrence coefficients of orthogonal polynomials using an
    algorithm that is provided as parameter.

    Parameters
    ----------
    nmom : int
        Number of moments.
    inv_type : type
        Subtype of `MomentInversion` that is  shall be tested.
    inv_setup : dict
        Setup dictionary with necessary parameters to initialize the selected
        algorithm.
    x_range : tuple
        Lower and upper bound for locations of quadrature nodes.

    """
    import numpy as np

    inv = inv_type(**inv_setup)

    # Generate weighted sum of Dirac-Delta densities and corresponding moments
    rng = np.random.default_rng(pytest.random_seed)
    xi0 = rng.uniform(low=x_range[0], high=x_range[1], size=nmom//2)
    xi0 = np.sort(xi0)
    w0 = rng.uniform(size=nmom//2)
    moments = np.array([np.dot(w0, xi0**k) for k in range(nmom)])

    # Moment inversion and check against original data
    xi, w = inv(moments)
    assert np.allclose(xi0, xi) and np.allclose(w0, w)

    # Test for odd number of moments
    if nmom > 2:
        a, b = inv.recurrence_coeffs(moments[:-1])
        assert len(b) == len(a) + 1


def _test_radau(inv_type, n_delta, n_nodes, inv_setup, x_range):
    """
    Test computation of Gauss-Radau quadrature with provided inversion algorithm
    using a weighted sum of Dirac-delta-densities and the Gauss-Radau-Laguerre
    formula from Example 3.5 in Ref. [:cite:label:`Gautschi_2004`].

    Parameters
    ----------
    n_delta : int
        Number of Delta-densities for the first test case.
    n_nodes : int
        Number of quadrature nodes, determines the number of required moments.
    inv_type : type
        Subtype of `MomentInversion` that is  shall be tested.
    inv_setup : dict
        Setup dictionary with necessary parameters to initialize the selected
        algorithm.
    x_range : tuple
        Lower and upper bound for locations of quadrature nodes.

    References
    ----------
        +-----------------+-----------------------+
        | [Gautschi_2004] | :cite:`Gautschi_2004` |
        +-----------------+-----------------------+

    """
    import numpy as np
    from scipy.special import gamma as gamma_func
    from scipy.special import genlaguerre
    from quadmompy.moments.special import gamma_moments

    rng = np.random.default_rng(pytest.random_seed)


    # Test case 1: Weighted sum of delta densities
    x = np.sort(rng.uniform(low=x_range[0], high=x_range[1], size=n_delta))
    w = rng.uniform(size=n_delta)
    nmom = 2*n_nodes - 1
    moments = np.array([np.dot(w, x**k) for k in range(nmom)])
    x0 = rng.uniform(low=min(x), high=max(x))
    inv_setup["radau"] = True
    inv_setup["radau_node"] = x0
    inv = inv_type(**inv_setup)
    x1, w1 = inv(moments)
    assert len(x1) == n_nodes
    assert np.any(np.isclose(x1, x0))
    moments_r = np.array([np.dot(w1, x1**k) for k in range(nmom)])


    # Test case 2: Gauss-Radau quadrature for generalized Laguerre measure, see
    # Example 3.5 in Ref. [Gautschi_2004]. Test for the "normal" case where the
    # fixed node is the lower boundary of the support, i.e. x0=0, as well as an
    # arbitrary positive value
    alpha = rng.lognormal() + 1
    #alpha = 0.
    for x0 in [0., rng.gamma(alpha)]:
        inv.radau_node = x0

        # Moments of the Laguerre weight are proportional to moments of the
        # Gamma-distribution with shape parameter k = alpha + 1 and scale
        # parameter theta = 1. The factor of proportionality is the integral of
        # the Laguerre weight function, see 'recurrence coefficient'
        # beta[0]=Gamma(1+alpha) in Table 1.1, Ref. [Gautschi_2004].
        mom = gamma_func(1 + alpha)*gamma_moments(nmom, k=alpha+1, theta=1)

        # Recurrence coefficents (reference values)
        b = np.zeros(n_nodes)
        b[0] = mom[0]
        n = np.arange(n_nodes)
        b[1:] = n[1:]*(n[1:] + alpha)       # Table 1.1 in Ref. [Gautschi_2004]
        a = np.ones(n_nodes)
        a[:-1] += 2*n[:-1] + alpha          # Table 1.1 in Ref. [Gautschi_2004]
        n_ = n_nodes - 1

        # Example 3.5 in [Gautschi_2004]
        a[-1] = x0 - \
                b[-1]*genlaguerre(n_- 1, alpha, monic=True)(x0) \
              / (genlaguerre(n_, alpha, monic=True)(x0))

        # Computation using moment inversion
        alpha_, beta_ = inv.recurrence_coeffs(mom)
        assert np.allclose(alpha_, a)
        assert np.allclose(beta_, b)
        x, w = inv.quad_from_rc(alpha_, beta_)
        assert np.any(np.isclose(x, x0))
        mom_r = np.vander(x, nmom, increasing=True).T@w
        assert np.allclose(mom, mom_r)


def _test_lobatto(inv_type, n_nodes, inv_setup):
    """
    Test computation of Gauss-Radau quadrature with algorithm of provided type
    using the Gauss-Lobatto-Jacobi formula from Example 3.8 in Ref.
    [:cite:label:`Gautschi_2004`].

    Parameters
    ----------
    n_delta : int
        Number of Delta-densities for the first test case.
    n_nodes : int
        Number of quadrature nodes, determines the number of required moments.
    inv_setup : dict
        Setup dictionary with necessary parameters to initialize the selected
        algorithm.

    References
    ----------
        +-----------------+-----------------------+
        | [Gautschi_2004] | :cite:`Gautschi_2004` |
        +-----------------+-----------------------+

    """
    import numpy as np
    from scipy.special import gamma as gamma_func
    from scipy.special import jacobi
    from quadmompy.moments.special import beta_moments
    from quadmompy.moments.transform import linear_transform

    rng = np.random.default_rng(pytest.random_seed)
    nmom = 2*(n_nodes - 1)

    inv_setup["lobatto"] = True
    inv_setup["lobatto_nodes"] = [0., 1.]
    inv = inv_type(**inv_setup)

    # Random log-noramally distributed parameters of the Jacobi polynomials
    alpha_jac = rng.lognormal()
    beta_jac = rng.lognormal()
    for ab in [np.array([-1, 1.]), np.sort(rng.uniform(low=-1, high=1, size=2))]:
        inv.lobatto_nodes = ab

        # Moments of the Jacobi weight are proportional to moments of the Beta-distribution with
        # parameters alpha = alpha_jac + 1 and beta = beta_jac + 1. The factor of proportionality is
        # the integral of the Jacobi weight function, see Table 1.1 in Ref. [Gautschi_2004]
        beta0 = 2**(alpha_jac + beta_jac + 1) \
              * gamma_func(alpha_jac + 1) * gamma_func(beta_jac + 1) \
              / gamma_func(alpha_jac + beta_jac + 1)
        mom = beta_moments(nmom, alpha=beta_jac+1, beta=alpha_jac+1)
        mom = linear_transform(mom, 2, -1)
        mom *= beta0

        # Recurrence coefficents (reference values) from Table 1.1 in [Gautschi_2004]
        k = np.arange(n_nodes)
        b = 4*k*(k + alpha_jac)*(k + beta_jac)*(k + alpha_jac + beta_jac)
        b /= (2*k + alpha_jac + beta_jac)**2
        b /= 2*k + alpha_jac + beta_jac + 1
        b /= 2*k + alpha_jac + beta_jac - 1
        b[0] = mom[0]
        a = beta_jac**2 - alpha_jac**2
        a /= 2*k + alpha_jac + beta_jac
        a /= 2*k + alpha_jac + beta_jac + 2

        n_ = n_nodes - 2
        p_jac = lambda x, n: jacobi(n, alpha_jac, beta_jac, monic=True)(x)
        A = np.array([ \
                        [p_jac(ab[0], n_ + 1), p_jac(ab[0], n_)], \
                        [p_jac(ab[1], n_ + 1), p_jac(ab[1], n_)], \
                    ])
        a[-1], b[-1] = np.linalg.solve(A, ab*A[:,0])

        # Computation using moment inversion
        alpha_, beta_ = inv.recurrence_coeffs(mom)
        assert np.allclose(alpha_, a)
        assert np.allclose(beta_, b)
        x, w = inv.quad_from_rc(alpha_, beta_)
        assert np.any(np.isclose(x, ab[0]))
        assert np.any(np.isclose(x, ab[1]))
        mom_r = np.vander(x, nmom, increasing=True).T@w
        assert np.allclose(mom, mom_r)
