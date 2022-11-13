"""
Demonstration using the example of the moment equations derived from the one-dimensional Fokker-Planck equation. More precisely, this is the Case 3 in Ref. [1] (Section 5.4) using the GaG-QMOM, 6 moments and the RK2SSP-AR scheme. The other results in Ref. [1], Section 5.4, can be reproduced by varying the number of moments `nmom`, the ODE integration scheme `integrate` (`rk2ssp` or `rk2ssp_ar`) and the QMOM type `qbmm_type` in `qbmm_setup.py` (`GaGQMOM` or `QMOM`).

References
----------
.. [1] M. Pütz, M. Pollack, C. Hasse, M. Oevermann "A Gauss/anti-Gauss quadrature method of moments applied to population balance equations with turbulence-induced nonlinear phase-space diffusion", J. Comput. Phys. 466 (2022) 111363.

"""
import numpy as np
from quadmompy import qbmm
from quadmompy.equations import FokkerPlanckEq1D
from quadmompy.equations.integrate_1d import rk2ssp_ar, rk2ssp

# Input filenames
qbmm_setup_file = "qbmm_setup.py"
mom0_file = "mom_ref.dat"               # exact solution as initial condition

# Output file name
mom_out_file = "mom.dat"

# Constants in Fokker-Planck equation
gamma = 330.
phi = 16.

# Number of moments
nmom = 6

# Parameters for ODE integration
tspan = [0., 1.5e-2]                    # time interval for solution
adaptive = True                         # enable adaptive step size control
dt = 1e-6                               # initial step size
dtmax = 1e-4                            # maximum step size


# Definition of coefficients in the
# Fokker-Planck equation
a = lambda v: -gamma*abs(v)*v           # first drift function
b = lambda v: 0.25*phi*phi*np.sign(v)   # second (noise-induced) drift function
sigma = lambda v: phi*abs(v)**0.5       # Noise function


# Load initial moments (exact steady
# solution with the constants above)
mom0 = np.genfromtxt(mom0_file)[:nmom]

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
                    save_all=True, \
                    dtmax=dtmax \
                  )

# Write results to file for analysis
header="t {0:s}".format(' '.join("mom[{0:d}]".format(k) for k in range(nmom)))
np.savetxt(mom_out_file, np.column_stack((t, mom)), header=header)
