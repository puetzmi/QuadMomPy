import abc
import numpy as np
from scipy.special import gamma as gamma_func
from scipy import stats
from quadmompy.moments.transform import rc2mom


class RandomMoments:
    """
    Base class for the generation of random moment sequences.

    Parameters
    ----------
    nmom : int
        Fixed length of moment sequences generated by the object.
    rng : numpy.random.Generator
        Random number generator. If not provided, `random_seed` must be passed to create a new random number generator.
    random_seed : int
        Must be provided if `rng` is `None` and is only used in that case to create a new random number generator.
    dtype : type
        Floating point type. The choice of this variable does not affect the sampling procedure of random quantities but only the computation of moments and return values.

    Attributes
    ----------
    nmom : int
        Fixed length of moment sequences generated by the object.
    n : int
        Length of the `beta` recurrence coefficients. The highest moment order is either `2n-1` if `nmom` is even or `2n-2` if `nmom` is odd.
    iodd : int
        Indicates if `nmom` is even (`iodd = 0`) or odd (`iodd = 1`).
    _rng : numpy.random.Generator
        Random number generator.
    dtype : type
        Floating point type. The choice of this variable does not affect the sampling procedure of random quantities but only the computation of moments and return values.

    """
    def __init__(self, nmom, rng, random_seed, dtype):
        self.nmom = nmom
        self.n = (nmom + 1)//2
        self.iodd = nmom % 2
        self._rng = rng
        if self._rng is None:
            if random_seed is None:
                msg = "RandomMoments requires either `rng` (random number generator) or random seed as parameter."
                raise ValueError(msg)
            self._rng = np.random.default_rng(random_seed)
        self.dtype = dtype

    @abc.abstractmethod
    def generate(self):
        """
        Generate random moment sequence of length `nmom`.

        """
        pass

    @abc.abstractmethod
    def pdf(self, mom):
        """
        Value of the probability density function at given point `mom`.

        Parameters
        ----------
        mom : array
            Moment sequence with length `nmom`.

        """
        pass

    def __call__(self):
        """
        Generate random moment sequence of length `nmom` (alternative call).

        """
        return self.generate()


class RandomHamburgerMoments(RandomMoments):
    """
    Class for the generation of random Hamburger moment sequences.

    The random moments are generated based on independently distributed recurrence coefficients of orthogonal polynomials and the resulting distribution in Hamburger moment space as described in Refs. [:cite:label:`Dette_2012`] and [:cite:label:`Dette_2016`], extended to moment sequences with odd numbers of elements.

    Parameters
    ----------
    nmom : int
        Fixed length of moment sequences generated by the object.
    gamma : array
        An array containing the `gamma`-parameters. The length depends on `nmom`.
    delta : array
        An array containing the `delta`-parameters. The length depends on `nmom`.
    rng : numpy.random.Generator, optional
        Random number generator. If not provided, `random_seed` must be passed to create a new random number generator.
    random_seed : int
        Must be provided if `rng` is `None` and is only used in that case to create a new random number generator.
    dtype : type, optional
        Floating point type. The choice of this variable does not affect the sampling procedure of random quantities but only the computation of moments and return values. Default value is `numpy.float64`.

    Attributes
    ----------
    nmom : int
        Fixed length of moment sequences generated by the object.
    n : int
        Length of the `beta` recurrence coefficients. The highest moment order is either `2n-1` if `nmom` is even or `2n-2` if `nmom` is odd.
    iodd : int
        Indicates if `nmom` is even (`iodd = 0`) or odd (`iodd = 1`).
    _rng : numpy.random.Generator
        Random number generator.
    dtype : type
        Floating point type. The choice of this variable does not affect the sampling procedure of random quantities but only the computation of moments and return values.
    gamma : array
        An array containing the `gamma`-parameters. The length depends on `nmom`.
    delta : array
        An array containing the `delta`-parameters. The length depends on `nmom`.
    _alpha_rv : list
        List containing a `scipy.stats.norm` distribution with the given parameters for each recurrence coefficient `alpha`.
    _beta_rv : list
        List containing a `scipy.stats.gamma` distribution with the given parameters for each recurrence coefficient `beta`.
    alpha : array
        An array to store first set of recurrence coefficients.
    beta : array
        An array to store second set of recurrence coefficients.

    References
    ----------
        +--------------+--------------------+
        | [Dette_2012] | :cite:`Dette_2012` |
        +--------------+--------------------+
        | [Dette_2016] | :cite:`Dette_2016` |
        +--------------+--------------------+

    """
    def __init__(self, nmom, gamma, delta, rng=None, random_seed=None, dtype=np.float64):
        super().__init__(nmom, rng=rng, random_seed=random_seed, dtype=dtype)

        # First check validity of given parameters based on the conditions in Ref. [Dette2012]
        if (len(gamma) != self.n - 1) or np.any(gamma <= -2*(self.n - np.arange(1, self.n))):
            msg = "Invalid parameter: `gamma` must satisfy `len(gamma) == n - 1` \
                    and `gamma[k] > 2*(n - k - 1)`."
            raise ValueError(msg)
        self.gamma = gamma

        if (len(delta) != 2*self.n - 1) or np.any(delta <= 0.):
            msg = "Invalid parameter: `delta` must satisfy `len(delta) == 2*n - 1` \
                    and `all(delta > 0)`."
            raise ValueError(msg)
        self.delta = delta

        # Initialize normal distributions for generation of alphas. SciPy uses common
        # parameter `sigma`, i.e. the standard deviation whereas the variance `sigma**2`
        # is used in Ref. [Dette2012]
        self._alpha_rv = [stats.norm(scale=(0.5/d)**0.5) for d in self.delta[:2*self.n-self.iodd-1:2]]
        #self._alpha_rv = self._alpha_rv[:len(self._alpha_rv)-self.iodd]

        # Initialize Gamma-distributions for generation of betas. SciPy uses the k-theta
        # parametrization of the Gamma-distribution -> theta[j] = 1/delta[2*j+1]
        shape = self.gamma + 2*self.n - 2*np.arange(1, self.n)
        scale = 1./self.delta[1::2]
        self._beta_rv = [stats.gamma(a=shape[i], scale=scale[i]) for i in range(len(shape))]

        # initialize recurrence coefficients with extreme or invalid values
        self.alpha = -1e300*np.ones(self.n - self.iodd, dtype=self.dtype)
        self.beta = -1e300*np.ones(self.n, dtype=self.dtype)

    def pdf(self, mom, inv):
        """
        Value of the probability density function at given point `mom` in Hamburger moment space.

        For simplicity, the distributions of the recurrence coefficients of orthogonal polynomials associated with `mom` are used here in conjunction with with the Jacobian determinant, which follows from Eqs. (2.10) and (2.11) in Ref. [:cite:label:`Dette_2012`]. The explicit form of the PDF is given in Eq. (2.19) in Ref. [:cite:label:`Dette_2012`].

        Parameters
        ----------
        mom : int
            Hamburger moment sequence of length `nmom`.
        inv : MomentInversion
            A moment inversion algorithm, i.e. an object of a subclass of `MomentInversion`, to compute the recurrence coefficients.

        Returns
        -------
        pdf : float
            Probability density at given point `mom`.

        References
        ----------
            +--------------+--------------------+
            | [Dette_2012] | :cite:`Dette_2012` |
            +--------------+--------------------+

        """
        if len(mom) != self.nmom:
            msg = "The size of the given moment set `mom` does not match `self.nmom`."
            raise ValueError(msg)

        alpha, beta = inv.recurrence_coeffs(mom)

        # Factor corresponging to alpha coefficients
        f_alpha = np.prod([self._alpha_rv[i].pdf(alpha[i]) for i in range(self.n - self.iodd)])
        # Factor corresponging to beta coefficients
        f_beta = np.prod([self._beta_rv[i-1].pdf(beta[i]) for i in range(1, self.n)])
        # Jacobian determinant, follows from Eqs. (2.10) and (2.11) in Ref. [Dette_2012]
        jac_det = np.prod(beta**(2*self.n - 2*np.arange(self.n) - 1 - self.iodd))

        return f_alpha * f_beta / jac_det

    def _gen_recurrence_coeffs(self):
        """
        Generate random recurrence coefficients of orthogonal polynoamials.

        The recurrence coefficients are independently distributed random variables: The coefficients `alpha` are normally distributed and the `beta` coefficients are Gamma-distributed, see Lemma 2.3 in Ref. [:cite:label:`Dette_2016`].

        Returns
        -------
        alpha : array
            First set of random recurrence coefficients of orthogonal polynomials.
        beta : array
            Second set of random recurrence coefficients of orthogonal polynomials.

        """
        alpha = np.array([rv.rvs(random_state=self._rng) for rv in self._alpha_rv], dtype=self.dtype)
        beta = np.array([rv.rvs(random_state=self._rng) for rv in self._beta_rv], dtype=self.dtype)
        return alpha, np.append(1., beta)

    def generate(self):
        """
        Generate a random Hamburger moment sequence by sampling random recurrence coefficients of orthogonal polynomials and subsequently computing the corresponding moment sequence.

        Returns
        -------
        mom : array
            A random Hamburger moment sequence of length `self.nmom`.

        References
        ----------
        +--------------+--------------------+
        | [Dette_2016] | :cite:`Dette_2016` |
        +--------------+--------------------+

        """
        alpha, beta = self._gen_recurrence_coeffs()
        self.alpha[:] = alpha
        self.beta[:] = beta
        mom = rc2mom(alpha, beta)
        return mom
