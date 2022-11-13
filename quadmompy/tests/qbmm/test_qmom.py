"""
Tests of different QMOM-types.

"""
import pytest


@pytest.mark.parametrize("n_delta,n_nodes", [(i,j) for i in range(2,7) for j in range(2,7) if j >= i])
def test_qmom_std(n_delta, n_nodes, x_range=(-1, 1), random_seed=None):
    """
    Basic test of the standard quadrature method of moments (QMOM) using moments of weighted sums of Dirac-Delta densities.

    Parameters
    ----------
    n_delta : int
        Number of Dirac-Delta-peaks.
    n_nodes : int
        Number of QMOM-nodes.
    x_range : tuple
        Limits of the location of Dirac-Delta functions.

    """
    import numpy as np
    from quadmompy import qbmm

    if random_seed is None:
        random_seed = pytest.random_seed

    # Generate test data
    rng = np.random.default_rng(random_seed)
    x = np.sort(rng.uniform(low=x_range[0], high=x_range[1], size=n_delta))
    w = rng.uniform(size=n_delta)
    n_moments = 2*n_nodes
    moments = np.array([np.dot(w, x**k) for k in range(n_moments)])

    # Setup QMOM
    setup = { \
            'n_dims': 1, \
            'qbmm_type': 'QMOM', \
            'qbmm_setup': { \
                'inv_type': 'Wheeler', \
                'inv_setup': {"radau": False}, \
                }, \
            }
    qmom = qbmm.new(**setup)

    # Compute and check against original data
    x1, w1 = qmom.moment_inversion(moments)
    x1_reduced = x1[w1 != 0.]
    w1_reduced = w1[w1 != 0.]
    assert np.allclose(w1_reduced, w) and np.allclose(x1_reduced, x)


@pytest.mark.parametrize("n_delta,n_nodes", [(i,j) for i in range(2,7) for j in range(2,7) if j >= i])
def test_qmom_radau(n_delta, n_nodes, x_range=(-1, 1), random_seed=None):
    """
    Basic test of the standard quadrature method of moments (QMOM) with Gauss-Radau quadrature, using moments of weighted sums of Dirac-Delta densities.

    Parameters
    ----------
    n_delta : int
        Number of Dirac-Delta-peaks.
    n_nodes : int
        Number of QMOM-nodes.
    x_range : tuple
        Limits of the location of Dirac-Delta functions.

    """
    import numpy as np
    from quadmompy import qbmm

    if random_seed is None:
        random_seed = pytest.random_seed

    # Generate test data
    rng = np.random.default_rng(random_seed)
    x = np.sort(rng.uniform(low=x_range[0], high=x_range[1], size=n_delta))
    w = rng.uniform(size=n_delta)
    n_moments = 2*n_nodes - 1
    moments = np.array([np.dot(w, x**k) for k in range(n_moments)])
    x0 = rng.uniform(low=min(x), high=max(x))   # fixed Radau-node

    # Setup Radau-QMOM
    setup = { \
            'n_dims': 1, \
            'qbmm_type': 'QMOM', \
            'qbmm_setup': { \
                'inv_type': 'Wheeler', \
                'inv_setup': {"radau": True, "radau_node": x0}, \
                }, \
            }
    qmom = qbmm.new(**setup)

    # Compute and check against original data
    x1, w1 = qmom.moment_inversion(moments)
    assert np.any(np.isclose(x1, x0))
    moments_r = np.array([np.dot(w1, x1**k) for k in range(n_moments)])
    assert np.allclose(moments_r, moments)

    # Compute passing different Radau-node dynamically
    x0 = rng.uniform(low=min(x), high=max(x))   # fixed Radau-node
    x1, w1 = qmom.moment_inversion(moments, radau_node=x0)
    assert np.any(np.isclose(x1, x0))
    moments_r = np.array([np.dot(w1, x1**k) for k in range(n_moments)])
    assert np.allclose(moments_r, moments)


@pytest.mark.parametrize("n_nodes_lower", range(1,10))
def test_gag_qmom(n_nodes_lower, x_range=(-1, 1), random_seed=None):
    """
    Basic test of the Gauss/anti-Gauss quadrature method of moments (GaG-QMOM), using moments of weighted sums of Dirac-Delta densities.

    Parameters
    ----------
    n_nodes : int
        Number of nodes of the lower-order Gaussian quadrature. The Anti-Gaussian nodes are then `n_nodes + 1` and the total number `2*n_nodes + 1`.
    x_range : tuple
        Limits of the location of Dirac-Delta functions.

    """
    import numpy as np
    from quadmompy import qbmm

    # Number of Dirac-Delta peaks
    n_delta = 10

    if random_seed is None:
        random_seed = pytest.random_seed

    # Generate test data
    rng = np.random.default_rng(random_seed)
    x = np.sort(rng.uniform(low=x_range[0], high=x_range[1], size=n_delta))
    w = rng.uniform(size=n_delta)
    n = n_nodes_lower
    n_moments = 2*(n + 1)
    moments = np.array([np.dot(w, x**k) for k in range(n_moments)])

    # Setup QMOM
    setup = { \
            'n_dims': 1, \
            'qbmm_type': 'GaGQMOM', \
            'qbmm_setup': { \
                'inv_type': 'Wheeler', \
                'inv_setup': {"radau": False,}, \
                }, \
            }
    qmom = qbmm.new(**setup)

    # Compute and check number of GaG-quadrature nodes and agreement with original moments
    x1, w1 = qmom.moment_inversion(moments)
    assert len(x1) == 2*n + 1 and len(w1) == 2*n + 1
    moments_r = np.array([np.dot(w1, x1**k) for k in range(n_moments)])
    assert np.allclose(moments_r, moments)
