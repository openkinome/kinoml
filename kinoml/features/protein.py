"""
Featurizers that mostly concern protein-based models
"""
from __future__ import annotations
from collections import Counter
import logging
import numpy as np

from .core import (
    ParallelBaseFeaturizer,
    BaseOneHotEncodingFeaturizer,
    OEBaseModelingFeaturizer,
    OEBaseKLIFSModelingFeaturizer,
)
from ..core.proteins import ProteinStructure
from ..core.systems import System, ProteinSystem


class AminoAcidCompositionFeaturizer(ParallelBaseFeaturizer):

    """
    Featurizes the protein using the composition of the residues
    in the binding site.
    """
    from ..core.proteins import AminoAcidSequence

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # Initialize a Counter object with 0 counts
    _counter = Counter(sorted(AminoAcidSequence.ALPHABET))
    for k in _counter.keys():
        _counter[k] = 0

    def _featurize_one(self, system: System) -> np.array:
        """
        Featurizes a protein using the residue count in the sequence

        Parameters
        ----------
        system: System
            The System to be featurized. Sometimes it will

        Returns
        -------
        list of array
            The count of amino acid in the binding site.
        """
        count = self._counter.copy()
        count.update(system.protein.sequence)
        sorted_count = sorted(count.items(), key=lambda kv: kv[0])
        return np.array([number for _, number in sorted_count])


class OneHotEncodedSequenceFeaturizer(BaseOneHotEncodingFeaturizer):

    """
    Featurize the sequence of the protein to a one hot encoding
    using the symbols in ``ALL_AMINOACIDS``.
    """
    from ..core.proteins import AminoAcidSequence

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    ALPHABET = AminoAcidSequence.ALPHABET

    def _retrieve_sequence(self, system: System) -> str:
        from ..core.proteins import AminoAcidSequence

        for comp in system.components:
            if isinstance(comp, AminoAcidSequence):
                return comp.sequence


class OEProteinStructureFeaturizer(OEBaseModelingFeaturizer):
    """
    Given systems with exactly one protein, prepare the protein structure by:

     - modeling missing loops
     - building missing side chains
     - mutations, if `uniprot_id` or `sequence` attribute is provided for the protein component
       (see below)
     - removing everything but protein and water
     - protonation at pH 7.4

    The protein component of each system must have a `pdb_id` or a `path` attribute specifying
    the protein structure to prepare.

     - `pdb_id`: A string specifying the PDB entry of interest, required if `path` not given.
     - `path`: The path to the structure file, required if `pdb_id` not given.

    Additionally, the protein component can have the following optional attributes to customize
    the protein modeling:

     - `name`: A string specifying the name of the protein, will be used for generating the
       output file name.
     - `chain_id`: A string specifying which chain should be used.
     - `alternate_location`: A string specifying which alternate location should be used.
     - `expo_id`: A string specifying a ligand bound to the protein of interest. This is especially
       useful if multiple proteins are found in one PDB structure.
     - `uniprot_id`: A string specifying the UniProt ID that will be used to fetch the amino acid
       sequence from UniProt, which will be used for modeling the protein. This will supersede the
       sequence information given in the PDB header.
     - `sequence`: An `AminoAcidSequence` object specifying the amino acid sequence that should be
       used during modeling the protein. This will supersede a given `uniprot_id` and the sequence
       information given in the PDB header.

    Parameters
    ----------
    loop_db: str
        The path to the loop database used by OESpruce to model missing loops.
    cache_dir: str, Path or None, default=None
        Path to directory used for saving intermediate files. If None, default location
        provided by `appdirs.user_cache_dir()` will be used.
    output_dir: str, Path or None, default=None
        Path to directory used for saving output files. If None, output structures will not be
        saved.
    """
    from MDAnalysis.core import universe

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    _SUPPORTED_TYPES = (ProteinSystem,)

    def _featurize_one(self, system: ProteinSystem) -> universe:
        """
        Prepare a protein structure.

        Parameters
        ----------
        system: ProteinSystem
            A system object holding a protein component.

        Returns
        -------
        : universe
            An MDAnalysis universe of the featurized system.
        """

        logging.debug("Preparing protein structure ...")
        design_unit = self._get_design_unit(system)

        logging.debug("Extracting design unit components ...")
        protein, solvent = self._get_components(design_unit)[:-1]

        if hasattr(system.protein, "sequence"):
            protein = self._process_protein(protein, system.protein.sequence)

        logging.debug("Assembling components ...")
        solvated_protein = self._assemble_components(protein, solvent)

        logging.debug("Updating pdb header ...")
        solvated_protein = self._update_pdb_header(
            solvated_protein,
            protein_name=system.protein.name
        )

        logging.debug("Writing results ...")
        file_path = self._write_results(
            solvated_protein,
            "_".join([
                f"{system.protein.name}",
                f"{system.protein.pdb_id if hasattr(system.protein, 'pdb_id') else system.protein.path.stem}",
                f"chain{getattr(system.protein, 'chain_id', None)}",
                f"altloc{getattr(system.protein, 'alternate_location', None)}"
            ])
        )

        logging.debug("Generating new MDAnalysis universe ...")
        structure = ProteinStructure.from_file(file_path)

        if not self.output_dir:
            logging.debug("Removing structure file ...")
            file_path.unlink()

        return structure


class OEKLIFSKinaseStructureFeaturizer(OEBaseKLIFSModelingFeaturizer):
    """
    Given systems with exactly one protein component, prepare the kinase protein structure by:

     - selecting a proper PDB structure according to KLIFS
     - modeling missing loops
     - building missing side chains
     - mutations, if `uniprot_id`, `klifs_kinase_id` or `sequence` attribute is provided for the
       protein component (see below)
     - removing everything but protein and water
     - protonation at pH 7.4

    The protein component of each system must have at least one of the following attributes
    specifying the kinase protein structure to prepare.

     - `pdb_id`: A string specifying the PDB entry of interest. This attribute will also be used
       to fetch the `uniprot_id` and the corresponding wild type `sequence`, which will be used
       for modeling unless a `sequence` attribute is given.
     - `uniprot_id`: A string specifying the UniProt ID of the kinase of interest. It will be used
       to select structures from KLIFS and to fetch the amino acid sequence from UniProt, which
       will be used for modeling the protein unless a `sequence` attribute is given.
     - `klifs_kinase_id`: A string specifying the KLIFS kinase ID of the kinase of interest. It
       will be used to select structures from KLIFS and to fetch the amino acid sequence from
       UniProt, which will be used for modeling the protein unless a `sequence` attribute is
       given.

    Additionally, the protein component can have the following optional attributes to customize
    the kinase structure selection and modeling:

     - `name`: A string specifying the name of the protein, will be used for generating the
       output file name.
     - `chain_id`: A string specifying which chain should be used.
     - `alternate_location`: A string specifying which alternate location should be used.
     - `expo_id`: A string specifying a ligand bound to the protein of interest. This is especially
       useful if multiple proteins are found in one PDB structure.
     - `sequence`: An `AminoAcidSequence` object specifying the amino acid sequence that should be
       used during modeling the protein. This will supersede a given `uniprot_id`,
       `klifs_kinase_id` and the sequence information given in the PDB header.
     - `dfg`: A string specifying the DFG conformation of interest ("in", "out", "out-like").
     - `ac_helix`: A string specifying the aC helix conformation of interest ("in", "out",
       "out-like").

    Parameters
    ----------
    loop_db: str
        The path to the loop database used by OESpruce to model missing loops.
    cache_dir: str, Path or None, default=None
        Path to directory used for saving intermediate files. If None, default location
        provided by `appdirs.user_cache_dir()` will be used.
    output_dir: str, Path or None, default=None
        Path to directory used for saving output files. If None, output structures will not be
        saved.
    """
    from MDAnalysis.core import universe

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    _SUPPORTED_TYPES = (ProteinSystem,)

    def _featurize_one(self, system: ProteinSystem) -> universe:
        """
        Prepare a kinase structure.

        Parameters
        ----------
        system: ProteinSystem
            A system object holding a protein component.

        Returns
        -------
        : universe
            An MDAnalysis universe of the featurized system.
        """

        logging.debug("Interpreting kinase of interest ...")
        self._interpret_kinase(system.protein)

        logging.debug("Selecting structure to model ...")
        if hasattr(system.protein, "pdb_id"):
            kinase_details = self._select_kinase_structure_by_pdb_id(system)
        else:
            kinase_details = self._select_kinase_structure_by_klifs_kinase_id(system)

        logging.debug("Adding attributes to protein component ...")
        system = self._add_kinase_attributes(system, kinase_details)

        logging.debug("Preparing protein structure ...")
        design_unit = self._get_design_unit(system)

        logging.debug("Extracting design unit components ...")
        protein, solvent = self._get_components(design_unit)[:-1]

        if hasattr(system.protein, "sequence"):
            protein = self._process_protein(protein, system.protein.sequence)

        logging.debug("Assembling components ...")
        solvated_protein = self._assemble_components(protein, solvent)

        logging.debug("Updating pdb header ...")
        solvated_protein = self._update_pdb_header(
            solvated_protein,
            protein_name=system.protein.name
        )

        logging.debug("Writing results ...")
        file_path = self._write_results(
            solvated_protein,
            "_".join([
                f"{system.protein.name}",
                f"{system.protein.pdb_id if hasattr(system.protein, 'pdb_id') else system.protein.path.stem}",
                f"chain{getattr(system.protein, 'chain_id', None)}",
                f"altloc{getattr(system.protein, 'alternate_location', None)}"
            ])
        )

        logging.debug("Generating new MDAnalysis universe ...")
        structure = ProteinStructure.from_file(file_path)

        if not self.output_dir:
            logging.debug("Removing structure file ...")
            file_path.unlink()

        return structure
