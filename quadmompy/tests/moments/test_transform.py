"""
Test of quadmompy.moments.transform-module.

"""
import pytest

@pytest.mark.parametrize("nmom", [i for i in range(2, 22, 2)])
def test_linear_transform(nmom):
    """
    Test linear moment transformation with weighted sum of Dirac-Delta functions.

    Parameters
    ----------
    nmom : int
        Number of moments.

    """
    import numpy as np
    from quadmompy.moments.transform import linear_transform
    rng = np.random.default_rng(pytest.random_seed)       # pseudo-random number generator
    a, b = rng.normal(size=2)               # scaling and shifting
    x = rng.normal(size=nmom)               # original normally distributed r.v.; size must be at least nmom//2
    w = rng.uniform(size=nmom)              # uniformly distributed weights
    w /= sum(w)                             # make density a regular pdf
    V = np.vander(x, nmom)[:,::-1]          # original Vandermonde matrix
    mu_orig = V.T@w                         # original moments
    t = a*x + b                             # transformed r.v.
    V = np.vander(t, nmom)[:,::-1]          # transformed Vandermonde matrix
    mu_trans = V.T@w                        # transformed moments
    assert(np.allclose(linear_transform(mu_orig, a, b), mu_trans))


@pytest.mark.parametrize("nmom", [i for i in range(2, 9)])
def test_mom_canonmom(nmom):
    """
    Test conversion of ordinary to canonical moments and vice versa using Beta-distribution, see Example 1.5.3 in Ref. [:cite:label:`Dette_1997`].

    Parameters
    ----------
    nmom : int
        Number of moments.

    References
    ----------
        +--------------+--------------------+
        | [Dette_1997] | :cite:`Dette_1997` |
        +--------------+--------------------+

    """
    import numpy as np
    from quadmompy.moments.transform import mom2canonmom, canonmom2mom
    from quadmompy.moments.special import beta_moments

    # Generate random parameters
    rng = np.random.default_rng(pytest.random_seed)
    a, b = rng.uniform(size=2)

    # Reference set of moments and canonical moments
    mref = beta_moments(nmom, a, b, canonical=False)
    pref = beta_moments(nmom, a, b, canonical=True)

    # Test transformation of ordinary moments -> canonical moments
    p = mom2canonmom(mref)
    assert(np.allclose(p, pref))

    # Test transformation of canonical moments -> ordinary moments
    m = canonmom2mom(pref)
    assert(np.allclose(m, mref))


@pytest.mark.parametrize("nmom", [i for i in range(2, 9)])
def test_rc_canonmom(nmom):
    """
    Test conversion of recurrence coefficients to canonical moments and vice versa using Beta distribution.

    Parameters
    ----------
    nmom : int
        Number of moments.

    References
    ----------
        +-----------------+-----------------------+
        | [Gautschi_2004] | :cite:`Gautschi_2004` |
        +-----------------+-----------------------+

    """
    import numpy as np
    from quadmompy.moments.transform import rc2canonmom, canonmom2rc
    from quadmompy.moments.special import beta_moments
    from scipy.special import gamma as gamma_func

    # Generate random parameters
    rng = np.random.default_rng(pytest.random_seed)
    a, b = rng.uniform(size=2)

    # Compute canonical moments
    pref = beta_moments(nmom, a, b, canonical=True)
    mref = beta_moments(nmom, a, b, canonical=False)

    # Compute recurrence coefficients resulting from Jacobi polynomials,
    # see Table 1.1 in Ref. [Gautschi_2004], with parameter substitution a_jac = b - 1,
    # b_jac = a - 1 and linear transformation x = (t+1)/2.
    a_jac = b - 1
    b_jac = a - 1
    scale = 0.5
    shift = 0.5
    k = np.arange((nmom+1)//2)
    beta_ref = 4*k*(k + a_jac)*(k + b_jac)*(k + a_jac + b_jac)
    beta_ref /= (2*k + a_jac + b_jac)**2
    beta_ref /= 2*k + a_jac + b_jac + 1
    beta_ref /= 2*k + a_jac + b_jac - 1
    beta_ref *= scale**2
    beta_ref[0] = 1.
    k = np.arange(nmom//2)
    alpha_ref = (b_jac**2 - a_jac**2) \
          / (2*k + a_jac + b_jac) \
          / (2*k + a_jac + b_jac + 2)
    alpha_ref = alpha_ref*scale + shift

    # Test computation of recurrence coefficients from canonical moments ...
    p = rc2canonmom(alpha_ref, beta_ref)
    assert(np.allclose(p, pref))

    # ... and the opposite direction
    alpha, beta = canonmom2rc(pref)
    assert(np.allclose(alpha, alpha_ref))
    assert(np.allclose(beta, beta_ref))


@pytest.mark.parametrize("nmom", [i for i in range(2,9)])
def test_zeta_canonmom(nmom):
    """
    Test conversion of continued-fraction coefficients to canonical moments and vice versa, using the quantities associated with the Beta distribution, see Example 1.5.3 in Ref. [:cite:label:`Dette_1997`].

    Parameters
    ----------
    nmom : int
        Number of moments.

    References
    ----------
        +--------------+--------------------+
        | [Dette_1997] | :cite:`Dette_1997` |
        +--------------+--------------------+

    """
    import numpy as np
    from quadmompy.moments.transform import zeta2canonmom, canonmom2zeta
    from quadmompy.moments.transform import canonmom2mom
    from quadmompy.moments.special import beta_moments

    # Generate random parameters
    rng = np.random.default_rng(pytest.random_seed)
    a, b = rng.uniform(size=2)

    # Compute reference set of canonical moments
    pref = beta_moments(nmom, a, b, canonical=True)

    # Continued fraction coefficients associated with a Beta distribution, see Eq. (1.5.10) in Ref. [Dette_1997]
    a, b = b - 1, a - 1     # parametrization in Ref. [Dette_1997]
    zeta_ref = np.zeros(nmom)
    n = nmom//2
    zeta_ref[0] = 1             # Eq. (1.4.6) in Ref. [Dette_1997]
    iodd = nmom % 2
    for m in range(1,n+iodd):
        zeta_ref[2*m-1] = (m + a + b)*(b + m)
        zeta_ref[2*m-1] /= (2*m + a + b - 1)*(2*m + a + b)
        zeta_ref[2*m] = (a + m)*m
        zeta_ref[2*m] /= (2*m + a + b)*(2*m + a + b + 1)
    if not iodd:
        m = n
        zeta_ref[2*m-1] = (m + a + b)*(b + m)
        zeta_ref[2*m-1] /= (2*m + a + b - 1)*(2*m + a + b)

    # Test computation of continued-fraction coefficients from canonical moments ...
    p = zeta2canonmom(zeta_ref)
    assert(np.allclose(p, pref))

    # ... and the opposite direction
    zeta = canonmom2zeta(pref)
    assert(np.allclose(zeta, zeta_ref))


@pytest.mark.parametrize("n", [i for i in range(2,9)])
def test_zeta_rc(n):
    """
    Test conversion of recurrence coefficients of orthogonal polynomials to continued-fraction coefficients and vice versa, using the Beta distribution, see Example 1.5.3 in Ref. [:cite:label:`Dette_1997`].

    Parameters
    ----------
    nmom : int
        Total number of coefficients.

    References
    ----------
        +-----------------+-----------------------+
        | [Dette_1997]    | :cite:`Dette_1997`    |
        +-----------------+-----------------------+
        | [Gautschi_2004] | :cite:`Gautschi_2004` |
        +-----------------+-----------------------+

    """
    import numpy as np
    from quadmompy.moments.transform import zeta2rc, rc2zeta

    # Generate random parameters
    rng = np.random.default_rng(pytest.random_seed)
    a_, b_ = rng.uniform(size=2)

    # Continued fraction coefficients associated with a Beta distribution, see Eq. (1.5.10) in Ref. [Dette_1997]
    a, b = b_ - 1, a_ - 1     # parametrization in Ref. [Dette_1997]
    zeta_ref = np.zeros(n)
    n2 = n//2
    zeta_ref[0] = 1             # Eq. (1.4.6) in Ref. [Dette_1997]
    iodd = n % 2
    for m in range(1,n2+iodd):
        zeta_ref[2*m-1] = (m + a + b)*(b + m)
        zeta_ref[2*m-1] /= (2*m + a + b - 1)*(2*m + a + b)
        zeta_ref[2*m] = (a + m)*m
        zeta_ref[2*m] /= (2*m + a + b)*(2*m + a + b + 1)
    if not iodd:
        m = n2
        zeta_ref[2*m-1] = (m + a + b)*(b + m)
        zeta_ref[2*m-1] /= (2*m + a + b - 1)*(2*m + a + b)

    # Compute recurrence coefficients resulting from Jacobi polynomials,
    # see Table 1.1 in Ref [Gautschi_2004], with parameter substitution a_jac = b - 1,
    # b_jac = a - 1 and linear transformation x = (t+1)/2.
    a_jac = b_ - 1
    b_jac = a_ - 1
    scale = 0.5
    shift = 0.5
    k = np.arange((n+1)//2)
    beta_ref = 4*k*(k + a_jac)*(k + b_jac)*(k + a_jac + b_jac)
    beta_ref /= (2*k + a_jac + b_jac)**2
    beta_ref /= 2*k + a_jac + b_jac + 1
    beta_ref /= 2*k + a_jac + b_jac - 1
    beta_ref *= scale**2
    beta_ref[0] = 1.
    k = np.arange(n//2)
    alpha_ref = (b_jac**2 - a_jac**2) \
          / (2*k + a_jac + b_jac) \
          / (2*k + a_jac + b_jac + 2)
    alpha_ref = alpha_ref*scale + shift


    # Test computation of recurrence coefficients from continued-fraction coefficients...
    alpha, beta = zeta2rc(zeta_ref)
    assert(np.allclose(alpha, alpha_ref))
    assert(np.allclose(beta, beta_ref))

    # ... and the opposite direction
    zeta = rc2zeta(alpha_ref, beta_ref)
    assert(np.allclose(zeta, zeta_ref))


@pytest.mark.parametrize("nmom", [i for i in range(2, 9)])
def test_rc2mom(nmom):
    """
    Test computation of moments from the associated recurrence coefficients of orthogonal polynomials, essentially a 'reverse moment inversion', using a Beta distribution

    Parameters
    ----------
    nmom : int
        Number of moments

    References
    ----------
        +-----------------+-----------------------+
        | [Gautschi_2004] | :cite:`Gautschi_2004` |
        +-----------------+-----------------------+

    """
    import numpy as np
    from quadmompy.moments.special import beta_moments
    from scipy.special import gamma as gamma_func
    from quadmompy.moments.transform import rc2mom

    # Generate random parameters
    rng = np.random.default_rng(pytest.random_seed)
    a, b = rng.uniform(size=2)

    # Compute canonical moments
    mref = beta_moments(nmom, a, b, canonical=False)

    # Compute recurrence coefficients resulting from Jacobi polynomials,
    # see Table 1.1 in Ref. [Gautschi_2004], with parameter substitution a_jac = b - 1,
    # b_jac = a - 1 and linear transformation x = (t+1)/2.
    a_jac = b - 1
    b_jac = a - 1
    scale = 0.5
    shift = 0.5
    k = np.arange((nmom+1)//2)
    beta_ref = 4*k*(k + a_jac)*(k + b_jac)*(k + a_jac + b_jac)
    beta_ref /= (2*k + a_jac + b_jac)**2
    beta_ref /= 2*k + a_jac + b_jac + 1
    beta_ref /= 2*k + a_jac + b_jac - 1
    beta_ref *= scale**2
    beta_ref[0] = 1.
    k = np.arange(nmom//2)
    alpha_ref = (b_jac**2 - a_jac**2) \
          / (2*k + a_jac + b_jac) \
          / (2*k + a_jac + b_jac + 2)
    alpha_ref = alpha_ref*scale + shift

    # Compute moments using transformation function
    m = rc2mom(alpha_ref, beta_ref)

    # Check against analytical moments
    assert(np.allclose(m, mref))


@pytest.mark.parametrize("n", [i for i in range(2, 9)])
def test_rc2ops(n):
    """
    Test computation of orthogonal polynomial system from recurrence coefficients using the shifted Legendre polynomials, see Table 1.1 in Ref. [:cite:label:`Gautschi_2004`].

    Parameters
    ----------
    n : int
        Polynomial degree.

    References
    ----------
        +-----------------+-----------------------+
        | [Gautschi_2004] | :cite:`Gautschi_2004` |
        +-----------------+-----------------------+

    """
    import numpy as np
    from scipy.special import sh_legendre
    from quadmompy.moments.transform import rc2ops

    # Compute recurrence coefficients of shifted Legendre
    # polynomials, see Table 1.1 in Ref. [Gautschi_2004]
    k = np.arange(n, dtype=float)
    alpha = 0.5*np.ones(n)
    beta = np.ones(n)
    beta[1:] = 0.25/(4 - k[1:]**-2)

    # Reference using SciPy's orthogonal polynomials
    pc_ref = sh_legendre(n, monic=True).coef[::-1]

    # Compute OPS using the `rc2ops`-function and compare
    # coefficients of nth-degree polynomial against reference
    pc = rc2ops(alpha, beta)
    assert(np.allclose(pc[-1], pc_ref))
