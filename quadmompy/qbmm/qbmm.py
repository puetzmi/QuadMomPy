#
# Copyright (c) 2022 Michele Puetz.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

"""
Module with base class for all quadrature-based moment methods (QBMMs) as well as QBMM-related convenience functions.

"""
import abc
from quadmompy.core import utils
from quadmompy.core.io import parse_setup


class Qbmm(metaclass=abc.ABCMeta):
    """
    Base class for quadrature-based moment methods (QBMMs).

    """
    @abc.abstractmethod
    def moment_inversion(self, mom):
        """
        Compute quadrature from a given set of moments.

        Parameters
        ----------
        mom : array_like
            A realizable moment set.

        Returns
        -------
        x : array
            Quadrature absicissas, `len(x)` depends on the specified method.
        w : array
            Quadrature weights with same length as `x`.

        """
        pass

    @classmethod
    def new(cls, qbmm_type, qbmm_setup, n_dims=None, **kwargs):
        """
        Create new instance of a specified `Qbmm`-subtype by calling the `new`-method implemented in the subclass.

        Parameters
        ----------
        qbmm_type : str or type
            Qbmm-subtype, either directly as type or as string or by `name`-variable implemented in subclass.
        qbmm_setup : dict
            Dictionary containing the parameters required to initialize `qbmm_type`-object.

        Returns
        -------
        new : Qbmm
            Instance of the specified subclass through the subclass's `new`-method.

        Raises
        ------
        ValueError
            If `qbmm_type` is not found.

        """
        if n_dims is not None:
            qbmm_setup["n_dims"] = n_dims   # n_dims is needed in qbmm_setup by multivariate classes
        try:
            if issubclass(qbmm_type, cls):
                return qbmm_type.new(qbmm_setup, **kwargs)
        except TypeError:
            subclasses = utils.get_all_subclasses(cls)
            qbmm_types = {}
            for scls in subclasses:
                try:
                    qbmm_types[scls.name] = scls
                except AttributeError:          # not all subclasses have the attribute `name`
                    pass
            try:
                return qbmm_types[qbmm_type].new(qbmm_setup, **kwargs)
            except KeyError:
                try:
                    qbmm_types = {scls.__name__: scls for scls in subclasses}
                    return qbmm_types[qbmm_type].new(qbmm_setup, **kwargs)
                except KeyError:
                    msg = "Unknown {0:s}-type `{1:s}`.".format(cls.__name__, qbmm_type)
                    msg += " Available types are: {0!s}.".format(list(qbmm_types.keys()))
                    raise ValueError(msg)

    @classmethod
    def from_dict(cls, setup_dict):
        """
        Create new instance of a `Qbmm`-subclass from dictionary, :meth:`new`-method.

        Parameters
        ----------
        setup_dict : dict
            Dictionary containing the parameters required by `Qbmm.new`.

        Returns
        -------
        new : Qbmm
            Instance of the subclass specified in `setup_dict`.

        """
        return cls.new(**setup_dict)

    @classmethod
    def from_file(cls, filename):
        """
        Create new instance of a Qbmm subclass (see :meth:`new`) from an input file that must contain either a Python-dictionary or an OpenFOAM-style dictionary, see :meth:`~quadmompy.core.io.parse_setup`.

        Parameters
        ----------
        filename : str
            Name of the input file.

        Returns
        -------
        new : Qbmm
            Instance of the subclass specified in the given file.

        """
        return cls.from_dict(parse_setup(filename))

    def __call__(self, mom):
        """
        Alternative way to call the method :meth:`moment_inversion`.

        """
        return self.moment_inversion(mom)


new = Qbmm.new
from_dict = Qbmm.from_dict
from_file = Qbmm.from_file
