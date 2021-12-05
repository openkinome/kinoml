"""
Featurizers that use SCHRODINGER software.

[WIP]
"""
import logging
from pathlib import Path
from typing import Union, Iterable

from .core import ParallelBaseFeaturizer
from ..core.systems import ProteinLigandComplex


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
    """
    from MDAnalysis.core.universe import Universe

    def __init__(
            self,
            cache_dir: Union[str, Path, None] = None,
            output_dir: Union[str, Path, None] = None,
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
        import MDAnalysis as mda

        from ..modeling.SCHRODINGERModeling import run_prepwizard
        from ..utils import LocalFileStorage

        logging.debug("Interpreting system ...")
        system_dict = self._interpret_system(system)

        complex_path = LocalFileStorage.featurizer_result(
            self.__class__.__name__,
            f"{system_dict['protein_name']}_{system_dict['ligand_name']}_complex",
            "pdb",
            self.output_dir,
        )

        run_prepwizard(
            schrodinger_directory=self.schrodinger,
            input_file=system_dict["protein_path"],
            output_file=complex_path,
            cap_termini=True,
            build_loops=True,
            sequence=system_dict["protein_sequence"],
            protein_pH="neutral",
            propka_pH=7.4,
            epik_pH=7.4,
            force_field="3",
        )

        # ToDo: select chain
        # ToDo: select alternate location
        # ToDo: select ligand
        # ToDo: delete expression tags

        logging.debug("Generating new MDAnalysis universe ...")
        try:
            structure = mda.Universe(complex_path, in_memory=True)
        except FileNotFoundError:
            return None

        if not self.save_output:
            complex_path.unlink()

        return structure

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

        logging.debug("Interpreting protein component ...")
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
                logging.debug(
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

        logging.debug("Interpreting ligand component ...")
        if hasattr(system.ligand, "name"):
            system_dict["ligand_name"] = system.ligand.name

        return system_dict
