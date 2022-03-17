"""
Featurizers that mostly concern protein-based models
"""
from __future__ import annotations
from collections import Counter
import logging
from typing import Union

import numpy as np

from .core import ParallelBaseFeaturizer, BaseOneHotEncodingFeaturizer, OEBaseModelingFeaturizer
from ..core.proteins import Protein, KLIFSKinase
from ..core.sequences import AminoAcidSequence
from ..core.systems import ProteinSystem, ProteinLigandComplex


logger = logging.getLogger(__name__)


class SingleProteinFeaturizer(ParallelBaseFeaturizer):
    """
    Provides a minimally useful ``._supports()`` method for all Protein-like featurizers.
    """

    _COMPATIBLE_PROTEIN_TYPES = (Protein, KLIFSKinase)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _supports(self, system: Union[ProteinSystem, ProteinLigandComplex]) -> bool:
        """
        Check that exactly one protein is present in the System
        """
        super_checks = super()._supports(system)
        proteins = [c for c in system.components if isinstance(c, self._COMPATIBLE_PROTEIN_TYPES)]
        return all([super_checks, len(proteins) == 1])


class AminoAcidCompositionFeaturizer(SingleProteinFeaturizer):

    """Featurizes the protein using the composition of the residues in the binding site."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # Initialize a Counter object with 0 counts
    _counter = Counter(sorted(AminoAcidSequence.ALPHABET))
    for k in _counter.keys():
        _counter[k] = 0

    def _featurize_one(
        self, system: Union[ProteinSystem, ProteinLigandComplex]
    ) -> Union[np.array, None]:
        """
        Featurizes a protein using the residue count in the sequence.

        Parameters
        ----------
        system: ProteinSystem or ProteinLigandComplex
            The System to be featurized.

        Returns
        -------
        : np.array or None
            The count of amino acids in the binding site.
        """
        count = self._counter.copy()
        try:
            sequence = system.protein.sequence
        except ValueError:  # e.g. erroneous uniprot_id in lazy instantiation
            return None
        count.update(system.protein.sequence)
        sorted_count = sorted(count.items(), key=lambda kv: kv[0])
        return np.array([number for _, number in sorted_count])


class OneHotEncodedSequenceFeaturizer(BaseOneHotEncodingFeaturizer, SingleProteinFeaturizer):

    """Featurizes the sequence of the protein to a one hot encoding."""

    ALPHABET = AminoAcidSequence.ALPHABET

    def __init__(self, sequence_type: str = "full", **kwargs):
        """
        Featurizes the sequence of the protein to a one hot encoding.

        Parameters
        ----------
        sequence_type: str, default=full
            The sequence to use for one hot encoding ('full', 'klifs_kinase' or 'klifs_structure').
        """
        if sequence_type not in ["full", "klifs_kinase", "klifs_structure"]:
            raise ValueError(
                "Only 'full', 'klifs_kinase' and 'klifs_structure' are supported sequence_types, "
                f"you provided {sequence_type}."
            )
        self.sequence_type = sequence_type
        if sequence_type != "full":
            self.ALPHABET += "-"  # add gap symbol for KLIFS sequence to ALPHABET
        super().__init__(**kwargs)  # update ALPHABET first

    def _retrieve_sequence(self, system: Union[ProteinSystem, ProteinLigandComplex]) -> str:
        try:
            if self.sequence_type == "full":
                sequence = system.protein.sequence
            elif self.sequence_type == "klifs_kinase":
                sequence = system.protein.kinase_klifs_sequence
            else:
                sequence = system.protein.structure_klifs_sequence
        except ValueError:  # e.g. erroneous uniprot_id in lazy instantiation
            return ""
        return sequence


class OEProteinStructureFeaturizer(OEBaseModelingFeaturizer, SingleProteinFeaturizer):
    """
    Given systems with exactly one protein, prepare the protein structure by:

     - modeling missing loops
     - building missing side chains
     - mutations, if `uniprot_id` or `sequence` attribute is provided for
       the protein component (see below)
     - removing everything but protein and water
     - protonation at pH 7.4

    The protein component of each system must be a `core.proteins.Protein`
    or a subclass thereof, must be initialized with toolkit='OpenEye' and
    give access to a molecular structure, e.g. via a pdb_id. Additionally,
    the protein component can have the following optional attributes to
    customize the protein modeling:

     - `name`: A string specifying the name of the protein, will be used for
        generating the output file name.
     - `chain_id`: A string specifying which chain should be used.
     - `alternate_location`: A string specifying which alternate location
        should be used.
     - `expo_id`: A string specifying a ligand bound to the protein of
       interest. This is especially useful if multiple proteins are found in
       one PDB structure.
     - `uniprot_id`: A string specifying the UniProt ID that will be used to
       fetch the amino acid sequence from UniProt, which will be used for
       modeling the protein. This will supersede the sequence information
       given in the PDB header.
     - `sequence`: A  string specifying the amino acid sequence in
       one-letter-codes that should be used during modeling the protein. This
       will supersede a given `uniprot_id` and the sequence information given
       in the PDB header.

    Parameters
    ----------
    loop_db: str
        The path to the loop database used by OESpruce to model missing loops.
    cache_dir: str, Path or None, default=None
        Path to directory used for saving intermediate files. If None, default
        location provided by `appdirs.user_cache_dir()` will be used.
    output_dir: str, Path or None, default=None
        Path to directory used for saving output files. If None, output
        structures will not be saved.
    use_multiprocessing : bool, default=True
        If multiprocessing to use.
    n_processes : int or None, default=None
        How many processes to use in case of multiprocessing. Defaults to
        number of available CPUs.
    """
    from MDAnalysis.core.universe import Universe

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _featurize_one(self, system: ProteinSystem) -> Union[Universe, None]:
        """
        Prepare a protein structure.

        Parameters
        ----------
        system: ProteinSystem
            A system object holding a protein component.

        Returns
        -------
        : Universe or None
            An MDAnalysis universe of the featurized system. None if no design unit was found.
        """
        from pathlib import Path

        import MDAnalysis as mda

        structure = self._read_protein_structure(system.protein)
        if structure is None:
            logger.warning(
                f"Could not read protein structure for {system.protein}, returning None!"
            )
            return None

        logging.debug("Preparing protein structure ...")
        design_unit = self._get_design_unit(
            structure=structure,
            chain_id=system.protein.chain_id if hasattr(system.protein, "chain_id") else None,
            alternate_location=system.protein.alternate_location
            if hasattr(system.protein, "alternate_location")
            else None,
            has_ligand=hasattr(system.protein, "expo_id"),
            ligand_name=system.protein.expo_id if hasattr(system.protein, "expo_id") else None,
            model_loops_and_caps=False if system.protein.sequence else True,
        )  # if sequence is given model loops and caps separately later
        if not design_unit:
            logging.debug("No design unit found, returning None!")
            return None

        logging.debug("Extracting design unit components ...")
        protein, solvent = self._get_components(
            design_unit=design_unit,
            chain_id=system.protein.chain_id if hasattr(system.protein, "chain_id") else None,
        )[:-1]

        if system.protein.sequence:
            first_id = 1
            if "construct_range" in system.protein.metadata.keys():
                first_id = int(system.protein.metadata["construct_range"].split("-")[0])
            protein = self._process_protein(
                protein_structure=protein,
                amino_acid_sequence=system.protein.sequence,
                first_id=first_id,
            )

        logging.debug("Assembling components ...")
        solvated_protein = self._assemble_components(protein, solvent)

        logging.debug("Updating pdb header ...")
        solvated_protein = self._update_pdb_header(
            solvated_protein, protein_name=system.protein.name
        )

        logging.debug("Writing results ...")
        file_path = self._write_results(
            solvated_protein,
            "_".join(
                [
                    info
                    for info in [
                        system.protein.name,
                        system.protein.pdb_id
                        if system.protein.pdb_id
                        else Path(system.protein.metadata["file_path"]).stem,
                        f"chain{system.protein.chain_id}"
                        if hasattr(system.protein, "chain_id")
                        else None,
                        f"altloc{system.protein.alternate_location}"
                        if hasattr(system.protein, "alternate_location")
                        else None,
                    ]
                    if info
                ]
            ),
        )

        logging.debug("Generating new MDAnalysis universe ...")
        structure = mda.Universe(file_path, in_memory=True)

        if not self.output_dir:
            logging.debug("Removing structure file ...")
            file_path.unlink()

        return structure
