"""
Featurizers that can only get applied to ProteinLigandComplexes or
subclasses thereof
"""
import logging

from .core import OEBaseModelingFeaturizer
from ..core.proteins import ProteinStructure
from ..core.systems import ProteinSystem, ProteinLigandComplex


class OEComplexFeaturizer(OEBaseModelingFeaturizer):
    """
    Given systems with exactly one protein and one ligand, prepare the complex structure by:

     - modeling missing loops
     - building missing side chains
     - mutations, if `uniprot_id` or `sequence` attribute is provided for the protein component
       (see below)
     - removing everything but protein, water and ligand of interest
     - protonation at pH 7.4

    The protein component of each system must have a `pdb_id` or a `path` attribute specifying
    the complex structure to prepare. Additionally the protein component can have the following
    optional attributes to customize the protein modeling:

     - `name`: A string specifying the name of the protein, will be used for generating the
       output file name.
     - `chain_id`: A string specifying which chain should be used.
     - `alternate_location`: A string specifying which alternate location should be used.
     - `uniprot_id`: A string specifying the UniProt ID that will be used to fetch the amino acid
       sequence from UniProt, which will be used for modeling the protein. This will supersede the
       sequence information given in the PDB header.
     - `sequence`: An `AminoAcidSequence` object specifying the amino acid sequence that should be
       used during modeling the protein. This will supersede a given `uniprot_id` and the sequence
       information given in the PDB header.

    The ligand component can be a BaseLigand without any further attributes. Additionally the
    ligand component can have the following optional attributes to customize the complex modeling:

     - `name`: A string specifying the name of the ligand, will be used for generating the
       output file name.
     - `expo_id`: A string specifying the ligand of interest. This is especially useful if
       multiple ligands are present in a PDB structure.

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

    Note
    ----
    If the ligand of interest is covalently bonded to the protein, the covalent bond will be
    broken. This may lead to the transformation of the ligand into a radical.
    """
    from MDAnalysis.core import universe

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    _SUPPORTED_TYPES = (ProteinLigandComplex,)

    def _featurize_one(self, system: ProteinLigandComplex) -> universe:
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

        logging.debug("Preparing complex structure ...")
        design_unit = self._get_design_unit(system)

        logging.debug("Extracting design unit components ...")
        protein, solvent, ligand = self._get_components(design_unit)

        if hasattr(system.protein, "sequence"):
            protein = self._process_protein(protein, system.protein.sequence)

        logging.debug("Assembling components ...")
        protein_ligand_complex = self._assemble_components(protein, solvent, ligand)

        logging.debug("Updating pdb header ...")
        protein_ligand_complex = self._update_pdb_header(
            protein_ligand_complex,
            protein_name=system.protein.name,
            ligand_name=system.ligand.name,
        )

        logging.debug("Writing results ...")
        file_path = self._write_results(
            protein_ligand_complex,
            "_".join([
                f"{system.protein.name}",
                f"{system.protein.pdb_id if hasattr(system.protein, 'pdb_id') else system.protein.path.stem}",
                f"chain{getattr(system.protein, 'chain_id', None)}",
                f"altloc{getattr(system.protein, 'alternate_location', None)}"
            ]),
            system.ligand.name,
        )

        logging.debug("Generating new MDAnalysis universe ...")
        structure = ProteinStructure.from_file(file_path)

        if not self.output_dir:
            logging.debug("Removing structure file ...")
            file_path.unlink()

        return structure
