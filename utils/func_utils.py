"""
This is used for some functions utils that we could use for the whole project.

"""
import warnings
import functools


def deprecated(func):
    """Used to deprecated some functions that we don't need, with a warning raised."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn("Called a function that we don't need:{}".format(func.__name__),
            category=DeprecationWarning, stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)
        return func(*args, **kwargs)
    return new_func

