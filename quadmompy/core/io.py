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
Module for input/output operations concerning moment sets and setup files.

"""
import numpy as np
import re


def read_moment_set(filename):
    """
    Read set of moments (or anything in suitable format) from a text file. The format must be
    either readable by the numpy.genfromtxt method (1D or 2D) or in the format
    .. (i,j,k,...) m[i,j,k,...],
    where (i,j,k,...) is the moment order in terms of a tuple of integers and m[i,j,k,...]
    the respective moment value.

    Parameters
    ----------
    filename : str

    Returns
    -------
    moments : array
        N-dimensional array of moments.

    """
    mom = np.genfromtxt(filename)
    check_fmt = ~np.any(np.isnan(mom)) # check if genfromtxt worked (only for 1D and 2D)
    if check_fmt:
        return mom

    # If NumPy-function did not work, read file line by line and parse
    # multidimensional indices and corresponging values
    dct = {}
    with open(filename, 'r') as fi:
        lines = fi.readlines()
        for line in lines: # read by index and value
            elmts = line.split()
            key = eval(''.join(elmts[:-1]))
            val = float(elmts[-1])
            dct[key] = val
    nmom = tuple(max(dim) + 1 for dim in zip(*dct.keys()))
    mom = np.zeros(nmom)
    indices = np.ndindex(*nmom)
    for idx in indices:
        mom[idx] = dct[idx]
    return mom


def write_moment_set(filename, mom, fmt='%.18e'):
    """
    Write set of moments (or any n-dimensional NumPy array) from a text file. Output is in
    array-format (1D and 2D) or in case of higher dimensionality `(i,j,k,...)  m[i,j,k,...]`, where `(i,j,k,...)` is the moment order in terms of a tuple of integers and `m[i,j,k,...]` the respective moment value.

    Parameters
    ----------
    filename : str
        Output filename.
    mom : array
        N-dimensional array of moments / real numbers.
    fmt : string, optional
        Format specifier

    """
    if len(mom.shape) < 3:
        np.savetxt(filename, mom, fmt=fmt)
    else:
        fmt = fmt[1:]
        with open(filename, 'w+') as fo:
            indices = np.ndindex(mom.shape)
            lines = []
            for idx in indices:
                lines.append("{0!s} {1:{2:s}}\n".format(idx, mom[idx], fmt))
            fo.writelines(lines)


def _convert_numbers_deep(data):
    """
    Convert all strings in a nested data structure to floats or ints if possible.

    Parameters
    ----------
    data : object
        Any data structure or object in general.

    Returns
    -------
    data : object
        Data structure with all elements that are 'numeric strings' converted to numeric data type.

    """
    if isinstance(data, str):
        try:
            x = float(data)
            if x == int(x):
                return int(x)
            return x
        except ValueError:
            return data
    if isinstance(data, dict):
        return {key: _convert_numbers_deep(val) for key, val in data.items()}
    if isinstance(data, list):
        return [_convert_numbers_deep(val) for val in data]
    if isinstance(data, tuple):
        return tuple([_convert_numbers_deep(val) for val in data])
    return data


def parse_setup(filename):
    """
    Parse setup file for QBMM.

    Parameters
    ----------
    filename : str
        Name of the input file.
    Returns
    -------
    setup : dict
        Dictionary necessary to initialize QBMM (if all required entries were given
        in the setup file).

    """
    with open(filename, 'r') as fin:
        setup = fin.read()

    # first try to read Python-dictionary directly from input file (recommended type of input)
    try:
        setup = eval(setup)

    # if that fails, parse input file in OpenFOAM-dictionary-like syntax
    except SyntaxError:
        # separate all parentheses and brackets from actual entries
        setup = '{ ' + re.sub(r"(\(|\)|\[|\]|\{|\})", r" \g<1> ", setup) + ' }'
        # make all whitespace a single space
        setup = ' '.join(setup.split())
        # remove possible spaces between dictionary values and semicolon
        setup = setup.replace(' ;', ';')
        # add colon after each dictionary key
        setup = re.sub(r"(\w) ", r"\g<1>: ", setup)
        # enclose all keys and values by single quotes
        setup = re.sub(r" (\w)", r" '\g<1>", setup)
        setup = re.sub(r"(\w)(;|:)", r"\g<1>'\g<2>", setup)
        # separate entries by comma
        setup = setup.replace(';', ',')

        # `setup` should now have proper Python-dictionary syntax
        setup = eval(setup)
        # convert all 'numeric strings' to numeric types
        setup = _convert_numbers_deep(setup)

    return setup
