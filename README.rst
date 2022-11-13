=============
**QuadMomPy**
=============

QuadMomPy is a library for all sorts of fun with moments, Gaussian quadratures, orthogonal polynomials and quadrature-based moment methods for the solution of spatially homogeneous population balance equations.


Installation
------------
The QuadMomPy package can be easily installed by
::

    pip install quadmompy

or from the project repository
::

    pip install -e .

A comprehensive documentation and additional examples are only available in the project repository on GitLab/GitHub. Use 

::

    git clone https://gitlab.com/puetzm/quadmompy.git

to clone the code from the GitLab repository or

::

    git clone https://github.com/puetzmi/quadmompy.git

to clone from the project mirror on GitHub.

The repository also includes a `Dockerfile` to run tests and use the QuadMomPy package in a docker image.


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


License
-------

Copyright (c) 2022 Michele Puetz.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
