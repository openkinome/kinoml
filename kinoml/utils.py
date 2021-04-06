from pathlib import Path
from itertools import zip_longest
from collections import defaultdict
from typing import Iterable, Callable, Any, Type
from importlib import import_module


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


class LocalFileStorage:

    """
    Generate standardized paths for storing and reading data locally.
    """

    from appdirs import user_cache_dir

    DIRECTORY = Path(user_cache_dir())

    @staticmethod
    def rcsb_structure_pdb(pdb_id, directory=DIRECTORY):
        file_path = directory / f"rcsb_{pdb_id}.pdb"
        return file_path

    @staticmethod
    def rcsb_ligand_sdf(pdb_id, chemical_id, chain, altloc, directory=DIRECTORY):
        file_path = directory / f"rcsb_{pdb_id}_{chemical_id}_{chain}_{altloc}.sdf"
        return file_path

    @staticmethod
    def rcsb_electron_density_mtz(pdb_id, directory=DIRECTORY):
        file_path = directory / f"rcsb_{pdb_id}.mtz"
        return file_path

    @staticmethod
    def klifs_ligand_mol2(structure_id, directory=DIRECTORY):
        file_path = directory / f"klifs_{structure_id}_ligand.mol2"
        return file_path

    @staticmethod
    def featurizer_result(featurizer_name, result_details, file_format, directory=DIRECTORY):
        file_path = directory / f"kinoml_{featurizer_name}_{result_details}.{file_format}"
        return file_path

    @staticmethod
    def pdb_smiles_json(directory=DIRECTORY):
        file_path = directory / "pdb_smiles.json"
        return file_path


class FileDownloader:

    """
    Download and store files locally.
    """

    @staticmethod
    def rcsb_structure_pdb(pdb_id):
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        download_file(url, LocalFileStorage.rcsb_structure_pdb(pdb_id))

    @staticmethod
    def rcsb_electron_density_mtz(pdb_id):
        url = f"https://edmaps.rcsb.org/coefficients/{pdb_id}.mtz"
        download_file(url, LocalFileStorage.rcsb_electron_density_mtz(pdb_id))


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


def download_file(url: str, path: str):
    """
    Download a file and save it locally.
    Parameters
    ----------
    url: str
        URL for downloading data.
    path: str
        Path to save downloaded data.
    """
    import requests

    response = requests.get(url)
    with open(path, "wb") as write_file:
        write_file.write(response.content)
        # TODO: check if successful, e.g. response.ok


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
    from subprocess import run

    print("Watermark")
    print("---------")
    w = WaterMark(get_ipython()).watermark("-d -n -t -i -z -u -v -h -m -g -w -iv")

    nvidiasmi = find_executable("nvidia-smi")
    if nvidiasmi:
        print()
        print("nvidia-smi")
        print("----------")
        result = run([nvidiasmi], capture_output=True)
        stdout = result.stdout.decode("utf-8").strip()
        if stdout:
            print("stdout:", stdout, sep="\n")
        stderr = result.stderr.decode("utf-8").strip()
        if stderr:
            print("stderr:", stderr, sep="\n")

    conda = find_executable("conda")
    if conda:
        print()
        print("conda info")
        print("----------")
        result = run([conda, "info", "-s"], capture_output=True)
        stdout = result.stdout.decode("utf-8").strip()
        if stdout:
            print(stdout, sep="\n")
        stderr = result.stderr.decode("utf-8").strip()
        if stderr:
            print("stderr:", stderr, sep="\n")

        print()
        print("conda list")
        print("----------")
        result = run([conda, "list"], capture_output=True)
        stdout = result.stdout.decode("utf-8").strip()
        if stdout:
            print(stdout, sep="\n")
        stderr = result.stderr.decode("utf-8").strip()
        if stderr:
            print("stderr:", stderr, sep="\n")


def collapsible(fn, *args, **kwargs):
    from ipywidgets import Output, Accordion

    out = Output()
    with out:
        fn(*args, **kwargs)
    acc = Accordion(children=[out])
    acc.set_title(0, "View output")
    acc.selected_index = None
    return acc


def fill_until_next_multiple(container, multiple_of: int, factory):
    """
    Fill `container` with instances of `factory` until its length
    reaches the next multiple of `multiple_of`.

    `container` gets modified in place and returned.
    """
    if isinstance(container, list):
        action = container.append
    elif isinstance(container, set):
        action = container.add
    else:
        raise TypeError("`container` must be an instance of list or set")

    for _ in range((multiple_of - (len(container) % multiple_of)) % multiple_of):
        action(factory())

    return container


def import_object(import_path: str):
    """
    Import an object using its full import path

    Parameters
    ----------
    import_path : str
        Full import path to object, like `kinoml.core.measurements.MeasurementType`.

    Returns
    -------
    object
    """
    if "." in import_path:
        module_str, obj_str = import_path.rsplit(".", 1)
        module = import_module(module_str)
        return getattr(module, obj_str)
    return import_module(import_path)