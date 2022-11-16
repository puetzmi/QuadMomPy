# pylint: disable=import-outside-toplevel
"""
Test module for the conditional quadrature of moments (CQMOM).

"""
import pytest

@pytest.mark.parametrize("ndims", range(2, 5))
def test_cqmom(ndims, random_seed=None):    # pylint: disable=too-many-locals
    """
    Test of the CQMOM that is carried out by simply constructing a conditional
    quadrature from pseudo-random numbers, computing the moments of the
    multidimensional Dirac-Delta distribution and checking if the CQMOM results
    are close to the original data.

    Parameters
    ----------
    ndims : int
        Number of dimensions.

    """
    import numpy as np
    from itertools import product as itprod
    from quadmompy.core.quadrature import ConditionalQuadrature
    from quadmompy.qbmm.multivariate import MultivariateQbmm

    rtol = 1e-6

    if random_seed is None:
        random_seed = pytest.random_seed    # pylint: disable=no-member

    rng = np.random.default_rng(random_seed)
    n_nodes = [3, 3, 2, 2]
    support = [np.sort(rng.uniform(low=-10, high=10, size=2)) for _ in range(ndims)]

    n_nodes = n_nodes[:ndims]
    support = support[:ndims]
    setup = {'n_dims': ndims, \
             'qbmm_type': 'CQMOM', \
             'qbmm_setup': {'config1d': []}}
    for _ in range(ndims):
        setup['qbmm_setup']['config1d'].append( \
                {'qbmm_type': 'QMOM', \
                 'qbmm_setup': \
                 {'inv_type': 'Wheeler', \
                  'inv_setup': \
                  {'adaptive': 0, 'rmin': 1e-8, 'eabs': 1e-3} \
                 }})
    #qmom = QbmmGen.from_dict(setup)
    qmom = MultivariateQbmm.new(**setup)

    quad = ConditionalQuadrature.empty(n_nodes)
    for i,nn in enumerate(n_nodes):
        nodes_dim = [list(range(n)) for n in n_nodes[:i]]
        nodes_dim = list(itprod(*nodes_dim))
        nodes_dim = [(i,) + el for el in nodes_dim]
        for nd in nodes_dim:
            quad.xi_cond[nd] = np.sort(rng.uniform(high=support[i][1], low=support[i][0], size=nn))
            quad.w_cond[nd] = np.sort(rng.uniform(size=nn))
            quad.w_cond[nd] /= sum(quad.w_cond[nd])
    quad.update()
    size = tuple(n*2 for n in n_nodes)
    mom = np.zeros(size)
    mom_orders = list(np.ndindex(mom.shape))
    for mo in mom_orders:
        mom[mo] = quad.reconstruct(mo)
    quad2 = qmom.moment_inversion(mom)
    mom_rec = np.zeros(mom.shape)
    err_max = [0., None]
    for mo in mom_orders:
        mom_rec[mo] = quad2.reconstruct(mo)
        if err_max[0] < abs((mom_rec[mo] - mom[mo])/mom[mo]):
            err_max[0] = abs((mom_rec[mo] - mom[mo])/mom[mo])
            err_max[1] = mo

    assert err_max[0] < rtol
