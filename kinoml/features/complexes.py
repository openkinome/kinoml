"""
Featurizers that can only get applied to ProteinLigandComplexes or
subclasses thereof
"""
from __future__ import annotations
from functools import lru_cache
from typing import Union

from appdirs import user_cache_dir

from .core import BaseFeaturizer
from ..core.ligands import FileLigand, SmilesLigand
from ..core.proteins import FileProtein, PDBProtein
from ..core.systems import ProteinLigandComplex
from ..docking.OpenEyeDocking import (
    chemgauss_docking,
    create_box_receptor,
    create_hybrid_receptor,
    hybrid_docking,
    resids_to_box,
)
from ..modeling.OpenEyePreparation import (
    has_ligand,
    prepare_complex,
    prepare_protein,
    read_electron_density,
    read_molecules,
    read_smiles,
    write_molecules,
)


class OpenEyesProteinLigandDockingFeaturizer(BaseFeaturizer):

    """
    Given a System with exactly one protein and one ligand,
    dock the ligand in the designated binding pocket.

    We assume that a smiles and file-based System object will be passed;
    this means we will have a System.components with FileProtein and
    FileLigand or SmilesLigand. The file itself could be a URL.
    """

    def __init__(self, loop_db: Union[str, None] = None):
        self.loop_db = loop_db

    _SUPPORTED_TYPES = (ProteinLigandComplex,)

    @lru_cache(maxsize=100)
    def _featurize(self, system: ProteinLigandComplex) -> ProteinLigandComplex:
        """
        Perform docking with OpenEye toolkits and thoughtful defaults.
        Parameters
        ----------
        system: ProteinLigandComplex
            A system object holding protein and ligand information.
        Returns
        -------
        protein_ligand_complex: ProteinLigandComplex
            The same system but with docked ligand.
        """
        if isinstance(system.ligand, SmilesLigand):
            ligands = [read_smiles(system.ligand.smiles)]
        else:
            ligands = read_molecules(system.ligand.path)
        protein = read_molecules(system.protein.path)[0]
        # TODO: electron density might be redundant here, if already used in separate protein preparation workflow
        if system.protein.electron_density_path is not None:
            electron_density = read_electron_density(
                system.protein.electron_density_path
            )
        else:
            electron_density = None

        # TODO: more sophisticated decision between hybrid and chemgauss docking
        if has_ligand(protein):

            prepared_protein, prepared_ligand = prepare_complex(
                protein, electron_density, self.loop_db
            )
            hybrid_receptor = create_hybrid_receptor(prepared_protein, prepared_ligand)
            docking_poses = hybrid_docking(hybrid_receptor, ligands)
        else:  # TODO: this is very kinase specific, should be more generic
            if isinstance(system.protein, PDBProtein):
                # TODO: check possibility to define design unit with residue (would consider electron density)
                prepared_protein = prepare_protein(protein, self.loop_db)
                klifs_pocket = PDBProtein.klifs_pocket(
                    system.protein.pdb_id
                )  # TODO: specify chain and altloc
                box_dimensions = resids_to_box(protein, klifs_pocket)
                box_receptor = create_box_receptor(protein, box_dimensions)
                docking_poses = chemgauss_docking(box_receptor, ligands)
            else:
                raise NotImplemented

        # TODO: where to store data
        protein_path = f"{user_cache_dir()}/{system.protein.name}.pdb"  # mmcif writing not supported by openeye
        write_molecules([prepared_protein], protein_path)
        file_protein = FileProtein(path=protein_path)

        ligand_path = (
            f"{user_cache_dir()}/{system.protein.name}_{system.ligand.name}.sdf"
        )
        write_molecules(docking_poses, ligand_path)
        file_ligand = FileLigand(path=ligand_path)
        protein_ligand_complex = ProteinLigandComplex(
            components=[file_protein, file_ligand]
        )
        # TODO: returns a ProteinLigandComplex with PDBProtein and SmilesLigand instead of FileProtein and FileLigand
        return protein_ligand_complex
