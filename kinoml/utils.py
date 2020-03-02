from pathlib import Path
from itertools import zip_longest
from collections import defaultdict

from appdirs import AppDirs

APPDIR = AppDirs(appname="kinoml", appauthor="openkinome")
PACKAGE_ROOT = Path(__file__).parent


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
    return PACKAGE_ROOT / "data" / path


def grouper(iterable, n, fillvalue=None):
    """
    Given an iterable, consume it in n-sized groups,
    filling it with fillvalue if needed
    """
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


class defaultdictwithargs(defaultdict):
    """
    A defaultdict that will create new values based on the missing value
    """

    def __init__(self, call):
        super().__init__(None)  # base class doesn't get a factory
        self.call = call

    def __missing__(self, key):  # called when key not in dict
        result = self.call(key)
        self[key] = result
        return result
