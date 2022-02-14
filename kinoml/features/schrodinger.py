"""
Featurizers that use SCHRODINGER software.

[WIP]
"""
import logging
from pathlib import Path
from typing import Union, Iterable

from .core import ParallelBaseFeaturizer
from ..core.systems import ProteinLigandComplex


logger = logging.getLogger(__name__)


class SCHRODINGERComplexFeaturizer(ParallelBaseFeaturizer):
    """
    Given systems with exactly one protein and one ligand, prepare the complex structure by:

     - modeling missing loops
     - building missing side chains
     - mutations, if `uniprot_id` or `sequence` attribute is provided for the protein component
       (see below)
     - removing everything but protein, water and ligand of interest
     - protonation at pH 7.4

    The protein component of each system must have a `pdb_id` or a `path` attribute specifying
    the complex structure to prepare.

     - `pdb_id`: A string specifying the PDB entry of interest, required if `path` not given.
     - `path`: The path to the structure file, required if `pdb_id` not given.

    Additionally, the protein component can have the following optional attributes to customize
    the protein modeling:
     - `name`: A string specifying the name of the protein, will be used for generating the
       output file name.
     - `chain_id`: A string specifying which chain should be used.
     - `alternate_location`: A string specifying which alternate location should be used.
     - `expo_id`: A string specifying the ligand of interest. This is especially useful if
       multiple ligands are present in a PDB structure.
     - `uniprot_id`: A string specifying the UniProt ID that will be used to fetch the amino acid
       sequence from UniProt, which will be used for modeling the protein. This will supersede the
       sequence information given in the PDB header.
     - `sequence`: A string specifying the amino acid sequence in single letter codes to be used
       during loop modeling and for mutations.

    The ligand component can be a BaseLigand without any further attributes. Additionally, the
    ligand component can have the following optional attributes:

     - `name`: A string specifying the name of the ligand, will be used for generating the
       output file name.

    Parameters
    ----------
    cache_dir: str, Path or None, default=None
        Path to directory used for saving intermediate files. If None, default location
        provided by `appdirs.user_cache_dir()` will be used.
    output_dir: str, Path or None, default=None
        Path to directory used for saving output files. If None, output structures will not be
        saved.
    max_retry: int, default=3
        The maximal number of attempts to try running the prepwizard step.
    """
    from MDAnalysis.core.universe import Universe

    def __init__(
            self,
            cache_dir: Union[str, Path, None] = None,
            output_dir: Union[str, Path, None] = None,
            max_retry: int = 3,
            **kwargs,
    ):
        from appdirs import user_cache_dir

        super().__init__(**kwargs)
        self.cache_dir = Path(user_cache_dir())
        if cache_dir:
            self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if output_dir:
            self.output_dir = Path(output_dir).expanduser().resolve()
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.save_output = True
        else:
            self.output_dir = Path(user_cache_dir())
            self.save_output = False
        self.max_retry = max_retry

    _SUPPORTED_TYPES = (ProteinLigandComplex,)

    def _pre_featurize(self, systems: Iterable[ProteinLigandComplex]) -> None:
        """
        Check that SCHRODINGER variable exists.
        """
        import os

        try:
            self.schrodinger = os.environ["SCHRODINGER"]
        except KeyError:
            raise KeyError("Cannot find the SCHRODINGER variable!")
        return

    def _featurize_one(self, system: ProteinLigandComplex) -> Union[Universe, None]:
        """
        Prepare a protein structure.

        Parameters
        ----------
        system: ProteinLigandComplex
            A system object holding a protein and a ligand component.

        Returns
        -------
        : Universe or None
            An MDAnalysis universe of the featurized system or None if not successful.
        """
        from ..utils import LocalFileStorage

        logger.debug("Interpreting system ...")
        system_dict = self._interpret_system(system)

        if system_dict["protein_sequence"]:
            system_dict["protein_path"] = self._preprocess_structure(
                pdb_path=system_dict["protein_path"],
                chain_id=system_dict["protein_chain_id"],
                alternate_location=system_dict["protein_alternate_location"],
                expo_id=system_dict["protein_expo_id"],
                sequence=system_dict["protein_sequence"],
            )

        prepared_structure_path = self._prepare_structure(
            system_dict["protein_path"], system_dict["protein_sequence"]
        )

        prepared_structure = self._postprocess_structure(
            pdb_path=prepared_structure_path,
            chain_id=system_dict["protein_chain_id"],
            alternate_location=system_dict["protein_alternate_location"],
            expo_id=system_dict["protein_expo_id"],
            sequence=system_dict["protein_sequence"],
        )

        if self.save_output:
            logging.debug("Saving results ...")
            complex_path = LocalFileStorage.featurizer_result(
                self.__class__.__name__,
                f"{system_dict['protein_name']}_{system_dict['ligand_name']}_complex",
                "pdb",
                self.output_dir,
            )
            prepared_structure.atoms.write(complex_path)

        return prepared_structure

    def _interpret_system(self, system: ProteinLigandComplex) -> dict:
        """
        Interpret the attributes of the given system components and store them in a dictionary.

        Parameters
        ----------
        system: ProteinSystem or ProteinLigandComplex
            The system to interpret.

        Returns
        -------
        : dict
            A dictionary containing the content of the system components.
        """
        from ..databases.pdb import download_pdb_structure
        from ..core.sequences import AminoAcidSequence

        system_dict = {
            "protein_name": None,
            "protein_pdb_id": None,
            "protein_path": None,
            "protein_sequence": None,
            "protein_uniprot_id": None,
            "protein_chain_id": None,
            "protein_alternate_location": None,
            "protein_expo_id": None,
            "ligand_name": None,
        }

        logger.debug("Interpreting protein component ...")
        if hasattr(system.protein, "name"):
            system_dict["protein_name"] = system.protein.name

        if hasattr(system.protein, "pdb_id"):
            system_dict["protein_path"] = download_pdb_structure(
                system.protein.pdb_id, self.cache_dir
            )
            if not system_dict["protein_path"]:
                raise ValueError(
                    f"Could not download structure for PDB entry {system.protein.pdb_id}."
                )
        elif hasattr(system.protein, "path"):
            system_dict["protein_path"] = Path(system.protein.path).expanduser().resolve()
        else:
            raise AttributeError(
                f"The {self.__class__.__name__} requires systems with protein components having a"
                f" `pdb_id` or `path` attribute."
            )
        if not hasattr(system.protein, "sequence"):
            if hasattr(system.protein, "uniprot_id"):
                logger.debug(
                    f"Retrieving amino acid sequence details for UniProt entry "
                    f"{system.protein.uniprot_id} ..."
                )
                system_dict["protein_sequence"] = AminoAcidSequence.from_uniprot(
                    system.protein.uniprot_id
                )

        if hasattr(system.protein, "chain_id"):
            system_dict["protein_chain_id"] = system.protein.chain_id

        if hasattr(system.protein, "alternate_location"):
            system_dict["protein_alternate_location"] = system.protein.alternate_location

        if hasattr(system.protein, "expo_id"):
            system_dict["protein_expo_id"] = system.protein.expo_id

        logger.debug("Interpreting ligand component ...")
        if hasattr(system.ligand, "name"):
            system_dict["ligand_name"] = system.ligand.name

        return system_dict

    def _preprocess_structure(
            self,
            pdb_path: Union[str, Path],
            chain_id: Union[str, None],
            alternate_location: Union[str, None],
            expo_id: Union[str, None],
            sequence: str,
    ):
        from MDAnalysis.core.universe import Merge

        from ..modeling.MDAnalysisModeling import (
            read_molecule,
            select_chain,
            select_altloc,
            remove_non_protein,
            delete_expression_tags,
            delete_short_protein_segments,
            delete_alterations,
            renumber_protein_residues,
        )
        from ..utils import LocalFileStorage, sha256_objects

        clean_structure_path = LocalFileStorage.featurizer_result(
            self.__class__.__name__,
            sha256_objects(
                [pdb_path, chain_id, alternate_location, expo_id, sequence]
            ),
            "pdb",
            self.cache_dir,
        )

        if not clean_structure_path.is_file():
            logger.debug("Cleaning structure ...")

            logger.debug("Reading structure from PDB file ...")
            structure = read_molecule(pdb_path)

            if chain_id:
                logger.debug(f"Selecting chain {chain_id} ...")
                structure = select_chain(structure, chain_id)

            if alternate_location:
                logger.debug(f"Selecting alternate location {alternate_location} ...")
                structure = select_altloc(structure, alternate_location)

            if expo_id:
                logger.debug(f"Selecting ligand {expo_id} ...")
                structure = remove_non_protein(structure, exceptions=[expo_id])

            logger.debug("Deleting expression tags ...")
            structure = delete_expression_tags(structure, pdb_path)

            logger.debug("Splitting protein and non-protein ...")
            protein = structure.select_atoms("protein")
            not_protein = structure.select_atoms("not protein")

            logger.debug("Deleting short protein segments ...")
            protein = delete_short_protein_segments(protein)

            logger.debug("Deleting alterations in protein ...")
            protein = delete_alterations(protein, sequence)

            logger.debug("Deleting short protein segments 2 ...")
            protein = delete_short_protein_segments(protein)

            logger.debug("Renumbering protein residues ...")
            protein = renumber_protein_residues(protein, sequence)

            logger.debug("Merging cleaned protein and non-protein ...")
            structure = Merge(protein.atoms, not_protein.atoms)

            logger.debug("Writing cleaned structure ...")
            structure.atoms.write(clean_structure_path)
        else:
            logger.debug("Found cached cleaned structure ...")

        return clean_structure_path

    def _prepare_structure(self, input_file, sequence):

        from ..modeling.SCHRODINGERModeling import run_prepwizard, mae_to_pdb
        from ..utils import LocalFileStorage, sha256_objects

        prepared_structure_path = LocalFileStorage.featurizer_result(
            self.__class__.__name__,
            sha256_objects([input_file, sequence]),
            "pdb",
            self.cache_dir,
        )

        if not prepared_structure_path.is_file():

            for i in range(self.max_retry):
                logger.debug(f"Running prepwizard trial {i + 1}...")
                mae_file_path = prepared_structure_path.rename(
                        prepared_structure_path.with_suffix('.mae')
                    )
                run_prepwizard(
                    schrodinger_directory=self.schrodinger,
                    input_file=input_file,
                    output_file=mae_file_path,
                    cap_termini=True,
                    build_loops=True,
                    sequence=sequence,
                    protein_pH="neutral",
                    propka_pH=7.4,
                    epik_pH=7.4,
                    force_field="3",
                )
                if prepared_structure_path.is_file():
                    mae_to_pdb(self.schrodinger, mae_file_path, prepared_structure_path)
                    break
        else:
            logger.debug("Found cached prepared structure ...")

        if not prepared_structure_path.is_file():
            logger.debug("Running prepwizard was not successful, returning None ...")
            return None

        return prepared_structure_path

    @staticmethod
    def _postprocess_structure(
            pdb_path,
            chain_id,
            alternate_location,
            expo_id,
            sequence,
    ):

        from ..modeling.MDAnalysisModeling import (
            read_molecule,
            select_chain,
            select_altloc,
            remove_non_protein,
            update_residue_identifiers
        )

        logger.debug("Loading prepared structure ...")
        prepared_structure = read_molecule(pdb_path)

        if not sequence:
            if chain_id:
                logger.debug(f"Selecting chain {chain_id} ...")
                prepared_structure = select_chain(prepared_structure, chain_id)
            if alternate_location:
                logger.debug(f"Selecting alternate location {alternate_location} ...")
                prepared_structure = select_altloc(prepared_structure, alternate_location)
            if expo_id:
                logger.debug(f"Selecting ligand {expo_id} ...")
                prepared_structure = remove_non_protein(prepared_structure, exceptions=[expo_id])

        logger.debug("Updating residue identifiers ...")
        prepared_structure = update_residue_identifiers(prepared_structure)

        return prepared_structure