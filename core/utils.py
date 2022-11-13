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
