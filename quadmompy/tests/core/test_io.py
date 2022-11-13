import pytest


@pytest.mark.parametrize("ndims", [1,2,3])
def test_read_write(ndims):
    """
    Test reading and writing `ndims`-dimensional moments sets.

    Parameters
    ----------
    ndims : int
        Number of dimensions.

    """
    import numpy as np
    import os
    from quadmompy.core import io

    rng = np.random.default_rng(pytest.random_seed)

    # Make sure file does not exist
    tmpfile = "tmp0.dat"

    try:
        while tmpfile in os.listdir(os.getcwd()):
            tmpfile = "tmp{0:d}.dat".format(rng.integers(low=0, high=2**20))

        dims = rng.integers(low=3, high=7, size=ndims)
        moments = rng.normal(size=dims)

        io.write_moment_set(tmpfile, moments)
        m = io.read_moment_set(tmpfile)
        assert(np.all(m == moments))                 # This should work with default precision

        # Also test NumPy format when moments-array is one- or two-dimensional
        if ndims < 3:
            np.savetxt(tmpfile, moments)
            m = io.read_moment_set(tmpfile)
            assert(np.all(m == moments))


    except Exception as e:
        os.remove(tmpfile)
        raise e

    os.remove(tmpfile)


@pytest.mark.parametrize("fname_suffix", ["qmom", "cqmom"])
def test_parse_setup(fname_suffix):
    """
    Test parsing of setup files `setup_*` (in the test directory) containing both Python and OpenFOAM-style dicitonaries.

    Paramters
    ---------
    fname_suffix : str
        Filename suffix, must match a file in the test directory.

    """
    from quadmompy.core.io import parse_setup
    s1 = parse_setup("setup_alt_{0:s}".format(fname_suffix ))
    assert(isinstance(s1, dict))
    s2 = parse_setup("setup_dict_{0:s}".format(fname_suffix ))
    assert(str(s1) == str(s2))
