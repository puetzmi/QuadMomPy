"""
Tests of the Qbmm-module.

"""
import pytest

def test_qbmm():
    """
    Some simple tests of `Qbmm` base class functions.

    """
    import numpy as np
    from quadmompy import qbmm
    from quadmompy.qbmm.univariate import Qmom
    from quadmompy.qbmm.multivariate import ConditionalQmom

    mom = np.genfromtxt("test_moments.dat")

    # QMOM
    # ----
    # Read setup from file in Python-dictionary format and invert moments
    qbmm_obj = qbmm.from_file("setup_alt_qmom")
    x1, w1 = qbmm_obj.moment_inversion(mom[0])
    # Read setup from file in OpenFOAM_dictionary format and ensure equality of results
    qbmm_obj = qbmm.from_file("setup_dict_qmom")
    x2, w2 = qbmm_obj(mom[0])                   # test call-operator
    assert(np.all(x1 == x2) and np.all(w1 == w2))

    # CQMOM input
    # -----------
    # Read setup from file in Python-dictionary format and invert moments
    qbmm_obj = qbmm.from_file("setup_alt_cqmom")
    quad = qbmm_obj.moment_inversion(mom)
    xi1, w1 = quad.xi, quad.w
    # Read setup from file in OpenFOAM_dictionary format and ensure equality of results
    qbmm_obj = qbmm.from_file("setup_dict_cqmom")
    quad = qbmm_obj.moment_inversion(mom)
    xi2, w2 = quad.xi, quad.w
    assert(np.all(x1 == x2) and np.all(w1 == w2))
