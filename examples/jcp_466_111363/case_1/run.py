# pylint:disable=R0801
"""
Demonstration using the example of the moment equations derived from the
one-dimensional Fokker-Planck equation. More precisely, this is the Case 1 (pure
Ito diffusion) in Ref. [1] (Section 5.2) using the GaG-QMOM, 8 moments and the
RK2SSP-AR scheme. The other results in Ref. [1], Section 5.2, can be reproduced
by varying the number of moments `nmom`, the ODE integration scheme `integrate`
(`rk2ssp` or `rk2ssp_ar`) and the QMOM type `qbmm_type` in `qbmm_setup.py`
(`GaGQMOM` or `QMOM`).

References
----------
.. [1] M. Pütz, M. Pollack, C. Hasse, M. Oevermann "A Gauss/anti-Gauss
       quadrature method of moments applied to population balance equations with
       turbulence-induced nonlinear phase-space diffusion", J. Comput. Phys. 466
       (2022) 111363.

"""
import numpy as np
from quadmompy import qbmm
from quadmompy.equations import FokkerPlanckEq1D
from quadmompy.equations.integrate_1d import rk2ssp_ar, rk2ssp  # pylint:disable=unused-import

# Input filenames
qbmm_setup_file = "qbmm_setup.py"
mom0_file = "mom_ref.dat"           # exact solution at t=0 as initial condition

# Output file name
mom_out_file = "mom.dat"

# Constants in Fokker-Planck equation
#gamma = 330.       # no drift term in the Langevin equation
phi = 16.

# Number of moments
nmom = 8

# Parameters for ODE integration;
# consistent with data in 'mom_ref.dat'
tspan = [0., 1e-3]                      # time interval for solution
adaptive = False                        # disable adaptive step size control
dt = 1e-6                               # fixed step size


# Definition of coefficients in the
# Fokker-Planck equation
a = np.zeros_like                       # no drift
b = np.zeros_like                       # no noise-induced drift
sigma = lambda v: phi*abs(v)**0.5       # noise function


# Load initial moments from exact solution at t=0
mom0 = np.genfromtxt(mom0_file)[0,1:nmom+1]

# Create QBMM-object from setup file
qbmm_obj = qbmm.from_file(qbmm_setup_file)

# Use absolutely realizable RK2SSP solver
integrate = rk2ssp_ar

# Create `FokkerPlanckEq`-object
fpe = FokkerPlanckEq1D( \
                        a=a, \
                        b=b, \
                        sigma=sigma, \
                        qbmm=qbmm_obj, \
                        integrate=integrate, \
                    )

# Solve equation on the interval `tspan`
mom, t = fpe.solve( \
                    mom0=mom0, \
                    t=tspan, \
                    dt=dt, \
                    adaptive=adaptive, \
                    save_all=True \
                  )

# Write results to file for analysis
momstr = ' '.join([f"mom[{k}]" for k in range(nmom)])
header = f"t {momstr}"
np.savetxt(mom_out_file, np.column_stack((t, mom)), header=header)
