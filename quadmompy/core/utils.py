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
Module for some helpful utilities used by other parts of the package.

"""
def get_all_subclasses(cls):
    """
    Get all subclasses of a given base class all the way down to the bottom of the class hierarchy.

    Parameters
    ----------
    cls : type
        Base class type.

    Returns
    -------
    all_subclasses : list
        A flat list of all subclasses with no hierarchy, regardless of the actual class hierarchy.

    """
    all_subclasses = []
    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))
    return all_subclasses
