# pylint: disable=import-outside-toplevel,too-many-arguments,too-many-locals
"""
Tests of different EQMOM-types.

"""
import pytest


def _test_eqmom(
    mom, sigma, support, qbmm_setup, sig_atol, sig_rtol, m_rtol, rng    # pylint:disable=unused-argument
):
    """
    Function performing the common KDF-independent steps of all EQMOM-tests.

    Parameters
    ----------
    mom : array
        Set of moments as the basis for computation and reference. In order to
        check correctness of the reconstructed moments, one additional moment
        must be provided, e.g. 8 moments are needed to test the 3-node EQMOM.
    sigma : float
        EQMOM parameter used to compute the input moments.
    support : tuple
        Support of the distribution, may be (semi-)infinite.
    qbmm : dict
        Dictionary with keywords passed to the EQMOM algorithm.
    sig_atol : float
        Absolute tolerance used to find sigma in EQMOM algorithm.
    sig_rtol : float
        Relative tolerance used to find sigma in EQMOM algorithm.
    m_rtol : float
        Relative tolerance used for comparison of original and reconstructed
        moments.
    rng : `numpy.random.Generator`
        Random number generator.

    """
    import numpy as np
    from scipy.integrate import quad
    from quadmompy import qbmm

    nmom = len(mom)

    # Make sure that EQMOM works with mom[0] != 1
    scale = rng.uniform()
    mom *= scale

    # Create EQMOM object
    eqmom = qbmm.new(**qbmm_setup)

    # Moment inversion and subsequent reconstruction from quadrature
    xi, w = eqmom.moment_inversion(mom)
    mom_reconst = np.array([np.dot(xi**k, w) for k in range(len(mom))])

    # Check correctness of NDF using numerical quadrature
    nsub = 200              # Maximum number of subdivisions
    mom_num = [quad(lambda x, k: x**k*eqmom.ndf(x), *support, limit=nsub, args=(k,))[0]
               for k in range(nmom)
            ]
    assert np.allclose(mom, mom_num)

    # As the solution may not be unique results can only be tested based on the moments
    mom_reconst = np.vander(xi, nmom, increasing=True).T@w
    assert np.allclose(mom, mom_reconst)


@pytest.mark.parametrize("nmom", [3, 5, 7, 9])
def test_gaussian_eqmom(nmom, sig_atol=1e-8, sig_rtol=1e-6, m_rtol=1e-6):
    """
    Test of the extended quadrature method of moments (EQMOM) with Gaussian KDFs.

    Parameters
    ----------
    nmom : int
        Number of moments.
    sig_atol : float
        Absolute tolerance used to find sigma in EQMOM algorithm.
    sig_rtol : float
        Relative tolerance used to find sigma in EQMOM algorithm.
    m_rtol : float
        Relative tolerance used for comparison of original and reconstructed moments.

    """
    import numpy as np
    from quadmompy.moments.special import normal_moments

    # Parameters
    mu_abs_max = 10.
    sigma_max = 3.

    # Generate test data
    n_alpha = (nmom - 1)//2
    rng = np.random.default_rng(pytest.random_seed) # pylint: disable=no-member
    mu = np.sort(-mu_abs_max + 2*mu_abs_max*rng.uniform(size=n_alpha))
    sigma = sigma_max*rng.uniform()
    weights = rng.uniform(size=n_alpha)
    weights /= sum(weights)
    mom_orig = np.array([normal_moments(nmom, mu=mu[i], sigma=sigma) for i in range(n_alpha)])
    mom_orig = mom_orig.T@weights

    # Setup EQMOM
    setup = { \
            'n_dims': 1, \
            'qbmm_type': 'EQMOM', \
            'qbmm_setup': { \
                'inv_type': 'Wheeler', \
                'inv_setup': {'cutoff': 1e-50}, \
                'type': 'gaussian', \
                'n_ab': 20, \
                'atol': sig_atol, \
                'rtol': sig_rtol, \
                }, \
            }

    # Run test
    _test_eqmom(mom_orig, sigma, (-np.inf, np.inf), setup, sig_atol, sig_rtol, m_rtol, rng)


@pytest.mark.parametrize("nmom", [3, 5, 7, 9])
def test_laplace_eqmom(nmom, sig_atol=1e-8, sig_rtol=1e-6, m_rtol=1e-6):
    """
    Test of the extended quadrature method of moments (EQMOM) with Laplace-KDFs.

    Parameters
    ----------
    nmom : int
        Number of moments.
    sig_atol : float
        Absolute tolerance used to find sigma in EQMOM algorithm.
    sig_rtol : float
        Relative tolerance used to find sigma in EQMOM algorithm.
    m_rtol : float
        Relative tolerance used for comparison of original and reconstructed moments.

    """
    import numpy as np
    from quadmompy.moments.special import laplace_moments

    # Parameters
    mu_abs_max = 10.
    sigma_max = 3.

    # Generate test data
    n_alpha = (nmom - 1)//2
    rng = np.random.default_rng(pytest.random_seed) #pylint:disable=no-member
    mu = np.sort(rng.uniform(low=-mu_abs_max, high=mu_abs_max, size=n_alpha))
    sigma = sigma_max*rng.uniform()
    weights = rng.uniform(size=n_alpha)
    weights /= sum(weights)
    mom_orig = np.array([laplace_moments(nmom, mu=mu[i], b=sigma) for i in range(n_alpha)])
    mom_orig = mom_orig.T@weights

    # Setup EQMOM
    setup = { \
            'n_dims': 1, \
            'qbmm_type': 'EQMOM', \
            'qbmm_setup': { \
                'inv_type': 'Wheeler', \
                'inv_setup': {'cutoff': 1e-50}, \
                'type': 'laplace', \
                'n_ab': 20, \
                'atol': sig_atol, \
                'rtol': sig_rtol, \
                }, \
            }

    # Run test
    _test_eqmom(mom_orig, sigma, (-np.inf, np.inf), setup, sig_atol, sig_rtol, m_rtol, rng)


@pytest.mark.parametrize("nmom", [3, 5, 7, 9])
def test_gamma_eqmom(nmom, sig_atol=1e-8, sig_rtol=1e-6, m_rtol=1e-6, random_seed=None):
    """
    Test of the extended quadrature method of moments (EQMOM) with Gamma-KDFs.

    Parameters
    ----------
    nmom : int
        Number of moments.
    sig_atol : float
        Absolute tolerance used to find sigma in EQMOM algorithm.
    sig_rtol : float
        Relative tolerance used to find sigma in EQMOM algorithm.
    m_rtol : float
        Relative tolerance used for comparison of original and reconstructed moments.

    """
    import numpy as np
    from quadmompy.moments.special import gamma_moments

    # Parameters
    mu_max = 10.
    sigma_max = 3.

    if random_seed is None:
        random_seed = pytest.random_seed    # pylint:disable=no-member

    # Generate test data
    n_alpha = (nmom - 1)//2
    rng = np.random.default_rng(random_seed)
    mu = np.sort(rng.uniform(low=0., high=mu_max, size=n_alpha))
    sigma = sigma_max*rng.uniform()
    weights = rng.uniform(size=n_alpha)
    weights /= sum(weights)
    mom_orig = np.array([gamma_moments(nmom, k=mu[i]/sigma, theta=sigma) for i in range(n_alpha)])
    mom_orig = mom_orig.T@weights

    # Setup EQMOM
    setup = { \
            'n_dims': 1, \
            'qbmm_type': 'EQMOM', \
            'qbmm_setup': { \
                'inv_type': 'Wheeler', \
                'inv_setup': {'cutoff': 1e-50}, \
                'type': 'gamma', \
                'n_ab': 20, \
                'atol': sig_atol, \
                'rtol': sig_rtol, \
                }, \
            }

    # Run test
    _test_eqmom(mom_orig, sigma, (0., np.inf), setup, sig_atol, sig_rtol, m_rtol, rng)


# TODO: test with larger sets of moments when Pigou-algorithm is more robust
#@pytest.mark.parametrize("nmom", [3,5,7,9])
@pytest.mark.parametrize("nmom", [3, 5])
def test_beta_eqmom(nmom, sig_atol=1e-8, sig_rtol=1e-6, m_rtol=1e-6, random_seed=None):
    """
    Test of the extended quadrature method of moments (EQMOM) with Beta-KDFs.

    Parameters
    ----------
    nmom : int
        Number of moments.
    sig_atol : float
        Absolute tolerance used to find sigma in EQMOM algorithm.
    sig_rtol : float
        Relative tolerance used to find sigma in EQMOM algorithm.
    m_rtol : float
        Relative tolerance used for comparison of original and reconstructed moments.

    """
    import numpy as np
    from quadmompy.moments.special import beta_moments

    # Parameters
    a = 0.
    b = 1.
    sigma_max = 3.

    if random_seed is None:
        random_seed = pytest.random_seed    # pylint:disable=no-member

    # Generate test data
    n_alpha = (nmom - 1)//2
    rng = np.random.default_rng(random_seed)
    mu = np.sort(rng.uniform(low=a, high=b, size=n_alpha))
    sigma = sigma_max*rng.uniform()
    weights = rng.uniform(size=n_alpha)
    weights /= sum(weights)
    mom_orig = np.array([
        beta_moments(nmom, alpha=mu[i]/sigma, beta=(1 - mu[i])/sigma) for i in range(n_alpha)
    ])
    mom_orig = mom_orig.T@weights
    # Setup EQMOM
    setup = { \
            'n_dims': 1, \
            'qbmm_type': 'EQMOM', \
            'qbmm_setup': { \
                'inv_type': 'Wheeler', \
                'inv_setup': {"cutoff": 1e-20, "beta_tol": 1e-15}, \
                'type': 'beta', \
                'n_ab': 20, \
                'atol': sig_atol, \
                'rtol': sig_rtol, \
                'bounds': (0., 1.), \
                }, \
            }

    # Run test
    _test_eqmom(mom_orig, sigma, setup['qbmm_setup']['bounds'],
                setup, sig_atol, sig_rtol, m_rtol, rng
            )
