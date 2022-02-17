import logging
from pathlib import Path
from typing import Union

from MDAnalysis.core.universe import Universe


logger = logging.getLogger(__name__)


def read_molecule(path: Union[str, Path], guess_bonds: bool = True) -> Universe:
    """
    Read a molecule from a file. Uses Biopython to support reading of the CIF format.

    Parameters
    ----------
    path: str, pathlib.Path
        Path to molecule file.
    guess_bonds: bool, default=True
        If bonds should be guessed by the van-der-Waals radius.
    
    Returns
    -------
    molecule: MDAnalysis.core.universe.Universe
        The MDAnalysis universe.
    """
    import copy

    import MDAnalysis as mda
    from MDAnalysis.topology.tables import vdwradii

    # add Maestro-like atom names to vdwradii dictionary needed for guessing bonds
    # e.g. Cl instead of CL
    vdwradii_new = copy.deepcopy(vdwradii)
    for key, value in vdwradii.items():
        if len(key) == 2:
            vdwradii_new[f"{key[0]}{key[1].lower()}"] = value

    path = str(Path(path).expanduser().resolve())
    suffix = path.split(".")[-1]
    if suffix == "cif":
        from tempfile import NamedTemporaryFile
        from Bio.PDB import MMCIFParser, PDBIO

        parser = MMCIFParser()
        structure = parser.get_structure("", path)
        with NamedTemporaryFile(suffix="pdb") as tempfile:
            io = PDBIO()
            io.set_structure(structure)
            io.save(tempfile.name)
            molecule = mda.Universe(
                tempfile.name, in_memory=True, dt=0, vdwradii=vdwradii_new, guess_bonds=guess_bonds
            )
    else:
        molecule = mda.Universe(
            path, in_memory=True, dt=0, vdwradii=vdwradii_new, guess_bonds=guess_bonds
        )

    return molecule
