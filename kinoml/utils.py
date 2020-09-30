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
    def rcsb_kinase_domain_pdb(pdb_id, directory=DIRECTORY):
        file_path = directory / f"rcsb_{pdb_id}_kinase_domain.pdb"
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

    return
