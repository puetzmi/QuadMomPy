"""
Test of the `quadmom.equations`-subpackage
"""
import pytest

def test_steady_fokker_planck():
    """
    This test makes sure that the results in Ref. [:cite:label:`Puetz_2022`] (Case 3) are reproduced. For simplicity, the setup with 6-moment GaG-QMOM and the standard RKSSP2 scheme is used, where the results should be exact. For additional information, see the `examples`-directory where the same case is used.

    References
    ----------
        +--------------+--------------------+
        | [Puetz_2022] | :cite:`Puetz_2022` |
        +--------------+--------------------+

    """
    import numpy as np
    from quadmompy import qbmm
    from quadmompy.equations import FokkerPlanckEq1D
    from quadmompy.equations.integrate_1d import rk2ssp

    nmom = 6
    mom0 = np.genfromtxt("mom_ref.dat")[:nmom]

    # Parameters
    gamma = 330.
    phi = 16.
    tspan = [0., 1.5e-2]
    adaptive = True
    dt = 1e-6
    dtmax = 1e-4
    integrate = rk2ssp

    a = lambda v: -gamma*abs(v)*v           # first drift function
    b = lambda v: 0.25*phi*phi*np.sign(v)   # second (noise-induced) drift function
    sigma = lambda v: phi*abs(v)**0.5       # Noise function

    qbmm_obj = qbmm.new(ndims=1, qbmm_type="GaGQMOM", qbmm_setup={"inv_type": "Wheeler"})

    fpe = FokkerPlanckEq1D( \
                            a=a, \
                            b=b, \
                            sigma=sigma, \
                            qbmm=qbmm_obj, \
                            integrate=integrate \
                        )

    mom, t = fpe.solve( \
                        mom0=mom0, \
                        t=tspan, \
                        dt=dt, \
                        adaptive=adaptive, \
                        save_all=True, \
                        dtmax=dtmax \
                      )

    # This setup should yield the exact steady-state solution, see Ref. [Puetz_2022]
    assert np.allclose(mom0, mom[-1])
