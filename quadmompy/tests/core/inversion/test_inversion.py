"""
Test functions for the `inversion`-module.

"""
import pytest


@pytest.mark.parametrize("nmom", [2,4,6,8,10])
def test_wheeler(nmom, x_range=(-1,1)):
    """
    Test moment inversion with Wheeler algorithm implemented in `wheeler.Wheeler`-class.

    Parameters
    ----------
    nmom : int
        Number of moments.
    x_range : tuple, optional
        Lower and upper bound for locations of quadrature nodes. Default is (-1,1).

    """
    from _test_inversion_common import _test_std
    from quadmompy.core.inversion.wheeler import Wheeler

    inv_type = Wheeler
    inv_setup = {}
    _test_std(nmom, inv_type, inv_setup, x_range)


@pytest.mark.parametrize("nmom", [2,4,6,8,10])
def test_pd(nmom, x_range=(-1,1)):
    """
    Test moment inversion with Golub-Welsch algorithm implemented in `gwa.GolubWelsch`-class.

    Parameters
    ----------
    nmom : int
        Number of moments.
    x_range : tuple, optional
        Lower and upper bound for locations of quadrature nodes. Default is (-1,1).

    """
    from _test_inversion_common import _test_std
    from quadmompy.core.inversion.pd import ProductDifference

    inv_type = ProductDifference
    inv_setup = {}
    _test_std(nmom, inv_type, inv_setup, x_range)


@pytest.mark.parametrize("nmom", [2,4,6,8,10])
def test_gwa(nmom, x_range=(-1,1)):
    """
    Test moment inversion with product-difference algorithm implemented in `pd.ProductDifference`-class.

    Parameters
    ----------
    nmom : int
        Number of moments.
    x_range : tuple, optional
        Lower and upper bound for locations of quadrature nodes. Default is (-1,1).

    """
    from _test_inversion_common import _test_std
    from quadmompy.core.inversion.gwa import GolubWelsch

    inv_type = GolubWelsch
    inv_setup = {}
    _test_std(nmom, inv_type, inv_setup, x_range)


@pytest.mark.parametrize("nmom", [2,4,6,8])
def test_wheeler_adaptive(nmom):
    """
    Test moment inversion with the adaptive Wheeler algorithm implemented in `wheeler.WheelerAdaptive`-class.

    Parameters
    ----------
    nmom : int
        Number of moments.

    """
    from quadmompy.core.inversion import BasicMomentInversion
    import numpy as np

    setup = {'eabs': 1e-8, 'rmin': 1e-8, 'cutoff': 1e-8}
    xi_range = [-10, 10]
    wheeler = BasicMomentInversion.new("WheelerAdaptive", **setup)
    rng = np.random.default_rng(pytest.random_seed)
    xi0 = rng.uniform(low=xi_range[0], high=xi_range[1], size=nmom//2)
    xi0 = np.sort(xi0)
    w0 = rng.uniform(size=nmom//2)
    moments = np.array([np.dot(w0, xi0**k) for k in range(nmom)])
    xi, w = wheeler(moments)
    assert (np.allclose(xi0, xi) and np.allclose(w0, w))
    # test adaptive procedure
    # violate weight criterion
    if nmom > 2:
        w0[0] = w0[1]*(wheeler.rmin - 1e-10)
        moments = np.array([np.dot(w0, xi0**k) for k in range(nmom)])
        xi, w = wheeler(moments)
        # check if number of nodes is reduced
        n_real = len(np.nonzero(w)[0])
        assert n_real < len(w0)
        moments_reconst = np.array([np.dot(w, xi**k) for k in range(2*n_real)])
        assert (np.allclose(moments[:2*n_real], moments_reconst))
    # violate node-distance criterion
    if nmom > 4:
        xi0[1] = xi0[0] + max(np.diff(xi[1:]))*(wheeler.eabs - 1e-12)
        moments = np.array([np.dot(w0, xi0**k) for k in range(nmom)])
        xi, w = wheeler(moments)
        # check if number of nodes is reduced
        n_real = len(np.nonzero(w)[0])
        assert n_real < len(w0)
        moments_reconst = np.array([np.dot(w, xi**k) for k in range(2*n_real)])
        assert (np.allclose(moments[:2*n_real], moments_reconst))


@pytest.mark.parametrize("nmom", [2,4,6,8])
def test_pd_adaptive(nmom):
    """
    Test moment inversion with the adaptive product-difference algorithm implemented in `pd.ProductDifferenceAdaptive`-class.

    Parameters
    ----------
    nmom : int
        Number of moments.

    """
    from quadmompy.core.inversion import BasicMomentInversion
    import numpy as np

    setup = {'eabs': 1e-8, 'rmin': 1e-8}
    xi_range = [-10, 10]
    pd = BasicMomentInversion.new("ProductDifferenceAdaptive", setup)
    rng = np.random.default_rng(pytest.random_seed)
    xi0 = rng.uniform(low=xi_range[0], high=xi_range[1], size=nmom//2)
    xi0 = np.sort(xi0)
    w0 = rng.uniform(size=nmom//2)
    moments = np.array([np.dot(w0, xi0**k) for k in range(nmom)])
    xi, w = pd(moments)
    assert (np.allclose(xi0, xi) and np.allclose(w0, w))
    # test adaptive procedure
    # violate weight criterion
    if nmom > 2:
        w0[0] = w0[1]*pd.rmin*0.5           # PD-algortihm is less robust and hence requires larger distance from threshold than e.g. Wheeler
        moments = np.array([np.dot(w0, xi0**k) for k in range(nmom)])
        xi, w = pd(moments)
        # check if number of nodes is reduced
        n_real = len(np.nonzero(w)[0])
        assert n_real < len(w0)
        moments_reconst = np.array([np.dot(w, xi**k) for k in range(2*n_real)])
        assert (np.allclose(moments[:2*n_real], moments_reconst))
    # The less robust PD-algorthm fails when the node criterion is violated -> no testing


@pytest.mark.parametrize("n_delta,n_nodes", [(i,j) for i in range(2,7) for j in range(2,7) if j <= i])
def test_wheeler_radau(n_delta, n_nodes, x_range=(-1, 1)):
    """
    Test computation of Gauss-Radau quadrature with Wheeler algorithm using a weighted sum of Dirac-delta-densities and the Gauss-Radau-Laguerre formula from Example 3.5 in Ref. [:cite:label:`Gautschi_2004`].

    Parameters
    ----------
    n_delta : int
        Number of Delta-densities for the first test case.
    n_nodes : int
        Number of quadrature nodes, determines the number of required moments.
    x_range : tuple, optional
        Lower and upper bound for locations of quadrature nodes. Default is (-1,1).

    References
    ----------
        +-----------------+-----------------------+
        | [Gautschi_2004] | :cite:`Gautschi_2004` |
        +-----------------+-----------------------+

    """
    from _test_inversion_common import _test_radau
    from quadmompy.core.inversion.wheeler import Wheeler

    inv_type = Wheeler
    inv_setup = {}
    _test_radau(inv_type, n_delta, n_nodes, inv_setup, x_range)


@pytest.mark.parametrize("n_delta,n_nodes", [(i,j) for i in range(2,7) for j in range(2,7) if j <= i])
def test_pd_radau(n_delta, n_nodes, x_range=(-1, 1)):
    """
    Test computation of Gauss-Radau quadrature with product-difference algorithm using a weighted sum of Dirac-delta-densities and the Gauss-Radau-Laguerre formula from Example 3.5 in Ref. [:cite:label:`Gautschi_2004`].

    Parameters
    ----------
    n_delta : int
        Number of Delta-densities for the first test case.
    n_nodes : int
        Number of quadrature nodes, determines the number of required moments.
    x_range : tuple, optional
        Lower and upper bound for locations of quadrature nodes. Default is (-1,1).

    References
    ----------
        +-----------------+-----------------------+
        | [Gautschi_2004] | :cite:`Gautschi_2004` |
        +-----------------+-----------------------+

    """
    from _test_inversion_common import _test_radau
    from quadmompy.core.inversion.pd import ProductDifference

    inv_type = ProductDifference
    inv_setup = {}
    _test_radau(inv_type, n_delta, n_nodes, inv_setup, x_range)


@pytest.mark.parametrize("n_delta,n_nodes", [(i,j) for i in range(2,8) for j in range(2,8) if j <= i])
def test_gwa_radau(n_delta, n_nodes, x_range=(-1, 1)):
    """
    Test computation of Gauss-Radau quadrature with Golub-Welsch algorithm using a weighted sum of Dirac-delta-densities and the Gauss-Radau-Laguerre formula from Example 3.5 in Ref. [:cite:label:`Gautschi_2004`].

    Parameters
    ----------
    n_delta : int
        Number of Delta-densities for the first test case.
    n_nodes : int
        Number of quadrature nodes, determines the number of required moments.
    x_range : tuple, optional
        Lower and upper bound for locations of quadrature nodes. Default is (-1,1).

    References
    ----------
        +-----------------+-----------------------+
        | [Gautschi_2004] | :cite:`Gautschi_2004` |
        +-----------------+-----------------------+

    """
    from _test_inversion_common import _test_radau
    from quadmompy.core.inversion.gwa import GolubWelsch

    inv_type = GolubWelsch
    inv_setup = {}
    _test_radau(inv_type, n_delta, n_nodes, inv_setup, x_range)


@pytest.mark.parametrize("n_nodes", range(5,7))
def test_wheeler_lobatto(n_nodes):
    """
    Test computation of Gauss-Lobatto quadrature with Wheeler algorithm using the Gauss-Lobatto-Jacobi formula from Example 3.8 in Ref. [:cite:label:`Gautschi_2004`].

    Parameters
    ----------
    n_nodes : int
        Number of quadrature nodes, determines the number of required moments.

    References
    ----------
        +-----------------+-----------------------+
        | [Gautschi_2004] | :cite:`Gautschi_2004` |
        +-----------------+-----------------------+

    """
    from quadmompy.core.inversion.wheeler import Wheeler
    from _test_inversion_common import _test_lobatto

    inv_type = Wheeler
    inv_setup = {}
    _test_lobatto(inv_type, n_nodes, inv_setup)


@pytest.mark.parametrize("n_nodes", range(5,7))
def test_pd_lobatto(n_nodes):
    """
    Test computation of Gauss-Lobatto quadrature with product-difference algorithm using the Gauss-Lobatto-Jacobi formula from Example 3.8 in Ref. [:cite:label:`Gautschi_2004`].

    Parameters
    ----------
    n_nodes : int
        Number of quadrature nodes, determines the number of required moments.

    References
    ----------
        +-----------------+-----------------------+
        | [Gautschi_2004] | :cite:`Gautschi_2004` |
        +-----------------+-----------------------+

    """
    from quadmompy.core.inversion.pd import ProductDifference
    from _test_inversion_common import _test_lobatto

    inv_type = ProductDifference
    inv_setup = {}
    _test_lobatto(inv_type, n_nodes, inv_setup)


@pytest.mark.parametrize("n_nodes", range(5,7))
def test_gwa_lobatto(n_nodes):
    """
    Test computation of Gauss-Lobatto quadrature with Golub-Welsch algorithm using the Gauss-Lobatto-Jacobi formula from Example 3.8 in Ref. [:cite:label:`Gautschi_2004`].

    Parameters
    ----------
    n_nodes : int
        Number of quadrature nodes, determines the number of required moments.

    References
    ----------
        +-----------------+-----------------------+
        | [Gautschi_2004] | :cite:`Gautschi_2004` |
        +-----------------+-----------------------+

    """
    from quadmompy.core.inversion.gwa import GolubWelsch
    from _test_inversion_common import _test_lobatto

    inv_type = GolubWelsch
    inv_setup = {}
    _test_lobatto(inv_type, n_nodes, inv_setup)
