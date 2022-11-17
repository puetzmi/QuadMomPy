"""
This script can be used to plot the results from `run.py`.

"""
import numpy as np
import matplotlib.pyplot as plt


# Indicates which moment shall be plotted
order = 2

# Input files
results_file = "mom.dat"
ref_file = "mom_ref.dat"

# load results and reference data
t, mom = np.genfromtxt(results_file)[:,[0, order+1]].T
ref = np.genfromtxt(ref_file)[order]    # (assuming the reference solution is steady and thus 1D)

# Setup plot and compare
fig = plt.figure()
keys = ['u', 'l']
ax = {key: fig.add_subplot(211 + i) for i,key in enumerate(keys)}
ax['u'].plot(t, mom, label="Numerical solution")
ax['u'].plot([t[0],t[-1]], [ref]*2, label="Exact solution") # (again assuming steady solution)
ax['u'].set_ylabel(f"$m_{{{order}}}$")
ax['u'].legend()
ax['l'].semilogy(t, abs(mom - ref))
ax['l'].set_ylabel("Abs. error")
ax['l'].set_xlabel("t")
for key in keys:
    ax[key].grid(which='both', ls=':', lw=0.5)

plt.show()
