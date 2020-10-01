from pathlib import Path
from itertools import zip_longest
from collections import defaultdict
from typing import Iterable, Callable, Any

from appdirs import AppDirs

APPDIR = AppDirs(appname="kinoml", appauthor="openkinome")
PACKAGE_ROOT = Path(__file__).parent


class FromDistpatcherMixin:
    @classmethod
    def _from_dispatcher(cls, value, handler, handler_argname, prefix):
        available_methods = [n[len(prefix) :] for n in cls.__dict__ if n.startswith(prefix)]
        if handler not in available_methods:
            raise ValueError(
                f"`{handler_argname}` must be one of: {', '.join(available_methods)}."
            )
        return getattr(cls, prefix + handler)(value)


def datapath(path: str) -> Path:
    """
    Return absolute path to a file contained in this package's `data`.

    Parameters:
        path: Relative path to file in `data`.
    Returns:
        Absolute path
    """
    return PACKAGE_ROOT / "data" / path


def grouper(iterable: Iterable, n: int, fillvalue: Any = None) -> Iterable:
    """
    Given an iterable, consume it in n-sized groups,
    filling it with fillvalue if needed.

    Parameters:
        iterable: list, tuple, str or anything that can be grouped
        n: size of the group
        fillvalue: last group will be padded with this object until
            `len(group)==n`
    """
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


class defaultdictwithargs(defaultdict):
    """
    A defaultdict that will create new values based on the missing value

    Parameters:
        call: Factory to be called on missing key
    """

    def __init__(self, call: Callable):
        super().__init__(None)  # base class doesn't get a factory
        self.call = call

    def __missing__(self, key):  # called when key not in dict
        result = self.call(key)
        self[key] = result
        return result


def seed_everything(seed=1234):
    import random
    import os

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def watermark():
    from IPython import get_ipython
    from watermark import WaterMark
    from distutils.spawn import find_executable
    from subprocess import check_output

    print("Watermark")
    print("---------")
    w = WaterMark(get_ipython()).watermark("-d -n -t -i -z -u -v -h -m -g -w -iv")

    nvidiasmi = find_executable("nvidia-smi")
    if nvidiasmi:
        print()
        print("nvidia-smi")
        print("----------")
        print(check_output([nvidiasmi], universal_newlines=True))

    conda = find_executable("conda")
    if conda:
        print()
        print("conda")
        print("-----")
        print(check_output([conda, "info", "-s"], universal_newlines=True))
        print(check_output([conda, "list"], universal_newlines=True))

