"""
Test of quadmompy.moments.special-module.

"""
import pytest


# Test for six (even) and seven (odd) number should suffice to ensure general correctness
@pytest.mark.parametrize("nmom", [6,7])
def test_uniform_moments(nmom):
    """
    Test correctness of up to sixth-order ordinary and canonical moments of a uniform distribution. The scipy.stats-module is used as a reference for ordinary moments and verified explicit formulas for the transformation into canonical moments.

    Parameters
    ----------
    nmom : int
        Number of moments.

    """
    import numpy as np
    from quadmompy.moments.special import uniform_moments
    from _canonical_moments_expl import ordinary_to_canonical
    from scipy.stats import uniform
    rng = np.random.default_rng(2022)
    a, b = [0., 1.]     # standard Hausdorff problem
    m_ref = np.array([uniform.moment(k, loc=a, scale=b-a) for k in range(nmom)])
    p_ref = ordinary_to_canonical(m_ref)
    m = uniform_moments(nmom, a=a, b=b)
    assert(np.allclose(m, m_ref))
    p = uniform_moments(nmom, a=a, b=b, canonical=True)
    assert(np.allclose(p, p_ref))


# Test for six (even) and seven (odd) number should suffice to ensure general correctness
@pytest.mark.parametrize("nmom", [6,7])
def test_beta_moments(nmom):
    """
    Test correctness of up to sixth-order ordinary and canonical moments of a beta distribution. The scipy.stats-module is used as a reference for ordinary moments and verified explicit formulas for the transformation into canonical moments.

    Parameters
    ----------
    nmom : int
        Number of moments.

    """
    import numpy as np
    from quadmompy.moments.special import beta_moments
    from _canonical_moments_expl import ordinary_to_canonical
    from scipy.stats import beta
    rng = np.random.default_rng(2022)
    a, b = rng.uniform(size=2)
    m_ref = np.array([beta.moment(k, a=a, b=b) for k in range(nmom)])
    p_ref = ordinary_to_canonical(m_ref)
    m = beta_moments(nmom, alpha=a, beta=b)
    assert(np.allclose(m, m_ref))
    p = beta_moments(nmom, alpha=a, beta=b, canonical=True)
    assert(np.allclose(p, p_ref))


@pytest.mark.parametrize("nmom", [10])
def test_beta_moments(nmom):
    """
    Test correctness of raw, central, and absolute moments of the normal distribution. The scipy.stats-module is used as a reference for raw and central moments. In order to verify the absolute moments the integrate.quad function is used (the numerical integral that is known to be correct should be sufficent to verify the correct implementation of the exact formulas).

    Parameters
    ----------
    nmom : int
        Number of moments.

    """
    import numpy as np
    from scipy.integrate import quad
    from quadmompy.moments.special import normal_moments
    from scipy.stats import norm
    rng = np.random.default_rng(2022)
    mu = rng.normal()
    sigma = 1 + rng.lognormal()
    rv = norm(loc=mu, scale=sigma)

    # Absloute raw moments
    integrand = lambda x, k: abs(x)**k * rv.pdf(x)
    m_abs_raw = np.array([ \
        quad(integrand, -np.inf, np.inf, args=(k,))[0] \
        for k in range(nmom) \
        ])
    assert(np.allclose(normal_moments(nmom, mu, sigma, absolute=True), m_abs_raw))

    # Absolute central moments
    integrand = lambda x, k: abs(x - mu)**k * rv.pdf(x)
    m_abs_central = np.array([ \
        quad(integrand, -np.inf, np.inf, args=(k,))[0] \
        for k in range(nmom) \
        ])
    assert(np.allclose(normal_moments(nmom, mu, sigma, central=True, absolute=True), m_abs_central))

    # Raw moments
    m_raw = np.array([rv.moment(k) for k in range(nmom)])
    assert(np.allclose(normal_moments(nmom, mu, sigma, central=False), m_raw))

    # Central moments
    rv = norm(loc=0, scale=sigma)
    m_central = np.array([rv.moment(k) for k in range(nmom)])
    assert(np.allclose(normal_moments(nmom, mu, sigma, central=True), m_central))


@pytest.mark.parametrize("nmom", [10])
def test_beta_moments(nmom):
    """
    Test correctness of raw and central moments of the Gamma distribution. The scipy.stats-module is used as a reference for raw and central moments. As no explicit formulas for central moments are available (other than recursive formulas, see Ref. [2] in `gamma_moments`, which is merely the linear transformation performed in the tested function), only the first few central moments are checked using known relations to statistical properties of the distribution (variance, skewness, kurtosis).

    Parameters
    ----------
    nmom : int
        Number of moments.

    """
    import numpy as np
    from scipy.stats import gamma
    from quadmompy.moments.special import gamma_moments
    rng = np.random.default_rng(2022)
    k = rng.lognormal()
    theta = 1 + rng.lognormal()
    rv = gamma(a=k, scale=theta)

    # First check correctness of raw moments
    m_raw_ref = np.array([rv.moment(j) for j in range(nmom)])
    m_raw = gamma_moments(nmom, k=k, theta=theta)
    assert(np.allclose(m_raw_ref, m_raw))

    # Check correctness of first central moments
    m_central = gamma_moments(nmom, k=k, theta=theta, central=True)
    # Zeroth central moment is 1
    assert(np.isclose(m_central[0], 1.))
    # First central moment is 0
    assert(np.isclose(m_central[1], 0.))
    # Second central moment is the variance
    assert(np.isclose(m_central[2], rv.stats('v')))
    # Third central moment is related to the skewness: Skew = m_central[3]/var**1.5
    assert(np.isclose(m_central[3], rv.stats('s')*m_central[2]**1.5))
    # Fourth central moment is related to excess kurtosis: Kurt = m_central[4]/var**2 - 3
    assert(np.isclose(m_central[4]/m_central[2]**2 - 3, rv.stats('k')))
