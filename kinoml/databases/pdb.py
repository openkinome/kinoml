import logging
from typing import Iterable

from appdirs import user_cache_dir


logger = logging.getLogger(__name__)


def smiles_from_pdb(ligand_ids: Iterable[str]) -> dict:
    """
    Retrieve SMILES of molecules defined by their PDB chemical identifier.

    Parameters
    ----------
    ligand_ids: iterable of str
        PDB chemical identifier.

    Returns
    -------
    ligands: dict
        Dictionary with PDB chemical identifier as keys and SMILES as values.
    """
    import json
    import math
    import requests
    import urllib

    ligand_ids = list(set(ligand_ids))
    ligands = {}
    base_url = "https://data.rcsb.org/graphql?query="
    n_batches = math.ceil(len(ligand_ids) / 50)  # request maximal 50 smiles at a time
    for i in range(n_batches):
        ligand_ids_batch = ligand_ids[i * 50: (i * 50) + 50]
        logger.debug(f"Batch {i}\n{ligand_ids_batch}")
        query = '{chem_comps(comp_ids:[' + \
                ','.join(['"' + ligand_id + '"' for ligand_id in ligand_ids_batch]) + \
                ']){chem_comp{id}rcsb_chem_comp_descriptor{SMILES_stereo}}}'
        response = requests.get(base_url + urllib.parse.quote(query))
        for ligand in json.loads(response.text)["data"]["chem_comps"]:
            try:
                ligands[ligand["chem_comp"]["id"]] = ligand[
                    "rcsb_chem_comp_descriptor"
                ]["SMILES_stereo"]
            except TypeError:
                # missing smiles entry
                pass

    return ligands


def download_pdb_structure(pdb_id, directory=user_cache_dir()):
    """
    Download a PDB structure. If the structure is not available in PDB format, it will be download
    in CIF format.

    Parameters
    ----------
    pdb_id: str
        The PDB ID of interest.
    directory: str or Path, default=user_cache_dir
        The directory for saving the downloaded structure.

    Returns
    -------
    : Path or False
        The path to the the downloaded file if successful, else False.
    """
    from pathlib import Path

    from ..utils import LocalFileStorage, FileDownloader

    directory = Path(directory)

    # check for structure in PDB format
    pdb_path = LocalFileStorage.rcsb_structure_pdb(pdb_id, directory)
    if not pdb_path.is_file():
        if FileDownloader.rcsb_structure_pdb(pdb_id, directory):
            return pdb_path
    else:
        return pdb_path

    # check for structure in CIF format
    cif_path = LocalFileStorage.rcsb_structure_cif(pdb_id, directory)
    if not cif_path.is_file():
        if FileDownloader.rcsb_structure_cif(pdb_id, directory):
            return cif_path
    else:
        return cif_path

    return False
