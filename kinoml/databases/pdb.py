import logging
from pathlib import Path
from typing import Iterable, Union

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
        ligand_ids_batch = ligand_ids[i * 50 : (i * 50) + 50]
        logger.debug(f"Batch {i}\n{ligand_ids_batch}")
        query = (
            "{chem_comps(comp_ids:["
            + ",".join(['"' + ligand_id + '"' for ligand_id in ligand_ids_batch])
            + "]){chem_comp{id}rcsb_chem_comp_descriptor{SMILES_stereo}}}"
        )
        response = requests.get(base_url + urllib.parse.quote(query))
        for ligand in json.loads(response.text)["data"]["chem_comps"]:
            try:
                ligands[ligand["chem_comp"]["id"]] = ligand["rcsb_chem_comp_descriptor"][
                    "SMILES_stereo"
                ]
            except TypeError:
                # missing smiles entry
                pass

    return ligands


def download_pdb_structure(
    pdb_id: str, directory: Union[str, Path] = user_cache_dir()
) -> Union[Path, bool]:
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
        logger.debug("Downloading PDB entry in PDB format ...")
        if FileDownloader.rcsb_structure_pdb(pdb_id, directory):
            return pdb_path
    else:
        return pdb_path

    # check for structure in CIF format
    cif_path = LocalFileStorage.rcsb_structure_cif(pdb_id, directory)
    if not cif_path.is_file():
        logger.debug("Downloading PDB entry in CIF format ...")
        if FileDownloader.rcsb_structure_cif(pdb_id, directory):
            return cif_path
    else:
        return cif_path
    logger.debug(f"Could not download PDB entry {pdb_id}.")
    return False


def download_pdb_ligand(
    pdb_id: str,
    chain_id: str,
    expo_id: str,
    smiles: str = "",
    directory: Union[str, Path] = user_cache_dir(),
) -> Union[Path, bool]:
    """
    Download a ligand co-crystallized to a PDB structure and save in SDF format. If a SMILES is
    provided, the connectivity and protonation will be adjusted accordingly.

    Parameters
    ----------
    pdb_id: str
        The PDB ID of interest.
    chain_id: str
        The chain ID of the ligand.
    expo_id: str
        The residue name of the ligand.
    smiles: str, default=""
        The smiles of the small molecule describing the connectivity and protonation of the
        ligand.
    directory: str or Path, default=user_cache_dir
        The directory for saving the downloaded structure.

    Returns
    -------
    : Path or False
        The path to the the processed ligand file in SDF format if successful, else False.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from ..utils import LocalFileStorage

    directory = Path(directory)
    sdf_path = LocalFileStorage.rcsb_ligand_sdf(
        pdb_id=pdb_id,
        chain_id=chain_id,
        expo_id=expo_id,
        altloc=None,
        directory=directory,
    )
    if sdf_path.is_file():
        logger.debug(
            f"Found cached ligand file for PDB entry {pdb_id}, chain {chain_id}, ligand {expo_id}."
        )
        return sdf_path

    pdb_path = download_pdb_structure(pdb_id=pdb_id, directory=directory)
    if not pdb_path:
        return False

    suffix = str(pdb_path).split(".")[-1]
    if suffix == "cif":
        cif_path = str(pdb_path)
        pdb_path = LocalFileStorage.rcsb_structure_pdb(
            pdb_id=f"{pdb_id}_chain{chain_id}", directory=directory
        )
        if not pdb_path.is_file():
            from Bio.PDB import MMCIFParser, PDBIO

            logger.debug("Converting CIF to PDB format ...")
            parser = MMCIFParser()
            try:
                structure = parser.get_structure("", cif_path)[0][chain_id]
            except KeyError:
                logger.debug(f"Could not find chain {chain_id} in CIF file!")
                return False
            io = PDBIO()
            io.set_structure(structure)
            io.save(str(pdb_path))

    logger.debug("Extracting ligand with RDKit ...")
    try:
        pdb_mol = Chem.MolFromPDBFile(str(pdb_path), sanitize=False)
        if pdb_mol is None:
            logger.debug(f"Could not read {pdb_path} with RDKit.")
            return False
        pdb_mol_chains = Chem.SplitMolByPDBChainId(pdb_mol)
        chain = pdb_mol_chains[chain_id]
        chain_residues = Chem.SplitMolByPDBResidues(chain)
        ligand = chain_residues[expo_id]
    except KeyError:
        logger.debug(
            f"Could not find ligand {expo_id} for chain {chain_id} in PDB entry {pdb_id}."
        )
        return False

    if smiles:
        logger.debug("Adjusting connectivity and protonation according to given SMILES ...")
        ligand = Chem.RemoveHs(ligand)
        reference_mol = Chem.MolFromSmiles(smiles)
        ligand = AllChem.AssignBondOrdersFromTemplate(reference_mol, ligand)
        ligand = Chem.AddHs(ligand, addCoords=True)

    logger.debug("Writing extracted ligand to SDF file ...")
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(ligand)

    return sdf_path
