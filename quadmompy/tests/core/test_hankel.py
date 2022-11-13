import pytest


@pytest.mark.parametrize("nmom", [10])      # only one sufficiently large number of moments should suffice
def test_HankelMatrix(nmom):
    """
    Test Hankel module.

    Parameters
    ----------
    nmom : int
        Number of moments.

    """
    import numpy as np
    from quadmompy.core.hankel import HankelMatrix

    atol = 1e-14
    rng = np.random.default_rng(pytest.random_seed)

    # Generate moments from a random weighted sum of Dirac-Delta densities. A
    # Hausdorff moment sequence is tested here as only that guarantees the
    # positive definiteness of all matrices tested here.
    x = rng.uniform(size=nmom)
    w = rng.uniform(size=nmom)
    mom = np.vander(x, nmom, increasing=True).T@w

    # Test correctness of matrix for even number of moments and kind='lower'
    H_even_lower = HankelMatrix.from_moments(mom, 'lower', check_pd=True)
    assert(np.all(H_even_lower[0] == mom[:nmom//2]))
    assert(np.all(H_even_lower[0] == H_even_lower[:,0]))
    assert(np.all(H_even_lower[1] == mom[1:nmom//2+1]))
    assert(np.all(H_even_lower[1] == H_even_lower[:,1]))
    assert(np.all(H_even_lower[-1] == mom[(nmom//2-1):-1]))
    assert(np.all(H_even_lower[-1] == H_even_lower[:,-1]))

    # Test correctness of matrix for even number of moments and kind='upper'
    H_even_upper = HankelMatrix.from_moments(mom, 'upper', check_pd=True)
    assert(np.all(H_even_upper[0] == mom[1:nmom//2+1]))
    assert(np.all(H_even_upper[0] == H_even_upper[:,0]))
    assert(np.all(H_even_upper[1] == mom[2:nmom//2+2]))
    assert(np.all(H_even_upper[1] == H_even_upper[:,1]))
    assert(np.all(H_even_upper[-1] == mom[nmom//2:]))
    assert(np.all(H_even_upper[-1] == H_even_upper[:,-1]))

    # Test correctness of determinant.
    det = np.linalg.det(H_even_upper())
    assert(abs(H_even_upper.det() - det) < atol)
    det = np.linalg.det(H_even_lower())
    assert(abs(H_even_lower.det() - det) < atol)

    # Test correctness of R-matrix
    R = H_even_lower.chol()
    assert(np.allclose(R.T@R, H_even_lower()))
    assert(np.all(np.triu(R) == R))

    # Test operators
    assert(np.all( \
                (H_even_lower-H_even_upper)() == H_even_lower._data - H_even_upper._data \
            ))    # also p.d. for Hausdorff moments
    assert(np.all( \
                (H_even_lower + H_even_upper)() == H_even_lower._data + H_even_upper.matrix() \
            ))
    assert(H_even_lower + H_even_upper == H_even_upper + H_even_lower)

    # Test correctness of matrix for odd number of moments
    H_odd_lower = HankelMatrix.from_moments(mom[:-1], 'lower', check_pd=True)
    assert(H_odd_lower == HankelMatrix.from_moments(mom[:-1], 'upper', check_pd=True))
    assert(H_odd_lower == H_even_lower)
    H_odd_upper = HankelMatrix.from_moments(mom[1:], 'upper', check_pd=True)
    assert(H_odd_upper == HankelMatrix.from_moments(mom[1:], 'lower', check_pd=True))
    assert(H_odd_upper == H_even_upper)

    # Test check of positive definiteness
    mom[-3] *= -1
    error = None
    try:
        HankelMatrix.from_moments(mom, 'lower', check_pd=True)
    except Exception as e:
        error = e
    assert(isinstance(error, ValueError))

    # Test update-function
    temp = H_even_lower.copy()
    H_even_lower.update(np.append(H_even_upper[0], H_even_upper[1:,-1]), kind='upper', check_pd=False)
    assert(H_even_lower == H_even_upper)
    H_even_lower = temp
