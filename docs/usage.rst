Usage
-----
A simple example of a moment inversion using the quadrature method of moments with the Wheeler algorithm to compute a Gaussian quadrature rule from a set of moments is given below.

.. code:: python

    import numpy as np
    from quadmompy import qbmm
    from quadmompy.qbmm.univariate import Qmom
    from quadmompy.core.inversion import Wheeler

    # Create Qmom object using the Wheeler
    # algorithm for moment inversion
    qmom = qbmm.new(qbmm_type=Qmom, qbmm_setup={'inv_type': Wheeler})

    # Nodes `x` and weights `w` of a weighted
    # sum of degenerate distributions
    x = np.array([-0.5, 0.1, 1., 1.4])
    w = np.array([0.15, 0.4, 0.4, 0.05])

    # Compute moments of the distribution
    n = len(x)
    moments = np.vander(x, 2*n).T[::-1]@w

    # Invert moments to obtain quadrature nodes and
    # weights, which should, in this particular
    # equal the nodes and weights defined above.
    x_quad, w_quad = qmom.moment_inversion(moments)
    print(f"x = {x_quad}")
    print(f"w = {w_quad}")

For more examples of how to use the numerous classes provided with this package, see the `tests` directory and the `examples` directory in the project repository on GitLab (`<https://gitlab.com/puetzm/quadmompy.git>`_) and the project mirror on GitHub(`<https://github.com/puetzmi/quadmompy.git>`_).
