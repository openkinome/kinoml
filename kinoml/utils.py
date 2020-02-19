from pathlib import Path
from itertools import zip_longest

_HERE = Path(__file__).parent


class FromDistpatcherMixin:
    @classmethod
    def _from_dispatcher(cls, value, handler, handler_argname, prefix):
        available_methods = [
            n[len(prefix) :] for n in cls.__dict__ if n.startswith(prefix)
        ]
        if handler not in available_methods:
            raise ValueError(
                f"`{handler_argname}` must be one of: {', '.join(available_methods)}."
            )
        return getattr(cls, prefix + handler)(value)


def datapath(path):
    """
    Return absolute path to a file contained in this package's ``data``.
    Parameters
    ----------
    path : str
        Relative path to file in ``data``.
    Returns
    -------
    str
        Absolute path
    """
    return _HERE / "data" / path


def grouper(iterable, n, fillvalue=None):
    """
    Given an iterable, consume it in n-sized groups,
    filling it with fillvalue if needed
    """
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)
