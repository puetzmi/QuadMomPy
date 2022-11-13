"""
Module for the correction of unrealizable moment sequences of different types.

"""
import numpy as np
from quadmompy.moments.transform import rc2mom


# TODO: Correct Hamburger moment sequence (particularly important for GaG-CQMOM)
def correct_hamburger(mom, rc=None, inv=None):
    r"""
    Correct unrealizable Hamburger moment sequence by modifying first negative recurrence coefficient :math:`\beta` and computing the corresponding moments.

    Parameters
    ----------
    mom : array
        Moment sequence to be corrected.
    rc : tuple, optional
        Tuple containing the two sets of recurrence coefficients of orthogonal polynomials corresponding to the given moment set.
    inv : MomentInversion, optional
        Basic moment inversion algorithm. Must be provided if recurrence coefficients are not given.

    Returns
    -------
    mom_corr : array
        Modified realizable moment set.

    Raises
    ------
    ValueError :
        If neither a basic inversion algorithm nor the recurrence coefficients are provided.

    """
    pass
