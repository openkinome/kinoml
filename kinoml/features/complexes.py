"""
Featurizers that can only get applied to ProteinLigandComplexes or
subclasses thereof
"""
import logging
from typing import Union

from .core import OEBaseModelingFeaturizer, ParallelBaseFeaturizer
from ..core.ligands import Ligand
from ..core.proteins import Protein, KLIFSKinase
from ..core.systems import ProteinLigandComplex


logger = logging.getLogger(__name__)


class SingleLigandProteinComplexFeaturizer(ParallelBaseFeaturizer):
    """
    Provides a minimally useful ``._supports()`` method for all ProteinLigandComplex-like
    featurizers.
    """

    _COMPATIBLE_PROTEIN_TYPES = (Protein, KLIFSKinase)
    _COMPATIBLE_LIGAND_TYPES = (Ligand,)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _supports(self, system: Union[ProteinLigandComplex]) -> bool:
        """
        Check that exactly one protein and one ligand is present in the System
        """
        super_checks = super()._supports(system)
        proteins = [c for c in system.components if isinstance(c, self._COMPATIBLE_PROTEIN_TYPES)]
        ligands = [c for c in system.components if isinstance(c, self._COMPATIBLE_LIGAND_TYPES)]
        return all([super_checks, len(proteins) == 1]) and all([super_checks, len(ligands) == 1])


class OEComplexFeaturizer(OEBaseModelingFeaturizer, SingleLigandProteinComplexFeaturizer):
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
     - `sequence`: An `AminoAcidSequence` object specifying the amino acid sequence that should be
       used during modeling the protein. This will supersede a given `uniprot_id` and the sequence
       information given in the PDB header.

    The ligand component can be a BaseLigand without any further attributes. Additionally, the
    ligand component can have the following optional attributes:

     - `name`: A string specifying the name of the ligand, will be used for generating the
       output file name.

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
    raise_errors: bool, default=False
        If False, will try to catch errors and return None, else featurization will stop when Error is raised.


    Note
    ----
    If the ligand of interest is covalently bonded to the protein, the covalent bond will be
    broken. This may lead to the transformation of the ligand into a radical.
    """
    from MDAnalysis.core.universe import Universe

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    _SUPPORTED_TYPES = (ProteinLigandComplex,)

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

        logger.debug("Preparing protein ligand complex ...")
        design_unit = self._get_design_unit(
            structure=structure,
            chain_id=system.protein.chain_id if hasattr(system.protein, "chain_id") else None,
            alternate_location=system.protein.alternate_location if hasattr(
                system.protein, "alternate_location"
            ) else None,
            has_ligand=True,
            ligand_name=system.protein.expo_id if hasattr(system.protein, "expo_id") else None,
            model_loops_and_caps=False if system.protein.sequence else True,
        )  # if sequence is given model loops and caps separately later
        if not design_unit:
            logger.debug("No design unit found, returning None!")
            return None

        logger.debug("Extracting design unit components ...")
        protein, solvent, ligand = self._get_components(
            design_unit=design_unit,
            chain_id=system.protein.chain_id if hasattr(system.protein, "chain_id") else None
        )

        if system.protein.sequence:
            first_id = 1
            if "construct_range" in system.protein.metadata.keys():
                first_id = int(system.protein.metadata["construct_range"].split("-")[0])
            protein = self._process_protein(
                protein_structure=protein,
                amino_acid_sequence=system.protein.sequence,
                first_id=first_id,
                ligand=ligand,
            )

        logger.debug("Assembling components ...")
        protein_ligand_complex = self._assemble_components(protein, solvent, ligand)

        logger.debug("Updating pdb header ...")
        protein_ligand_complex = self._update_pdb_header(
            protein_ligand_complex,
            protein_name=system.protein.name,
            ligand_name=system.ligand.name,
        )

        logger.debug("Writing results ...")
        file_path = self._write_results(
            protein_ligand_complex,
            "_".join([info for info in [
                system.protein.name,
                system.protein.pdb_id if system.protein.pdb_id
                else Path(system.protein.metadata["file_path"]).stem,
                f"chain{system.protein.chain_id}" if hasattr(system.protein, "chain_id")
                else None,
                f"altloc{system.protein.alternate_location}"
                if hasattr(system.protein, "alternate_location") else None,
            ] if info]),
            system.ligand.name,
        )

        logger.debug("Generating new MDAnalysis universe ...")
        structure = mda.Universe(file_path, in_memory=True)

        if not self.output_dir:
            logger.debug("Removing structure file ...")
            file_path.unlink()

        return structure


class OEFredDockingFeaturizer(OEBaseModelingFeaturizer, SingleLigandProteinComplexFeaturizer):
    """
    Given systems with exactly one protein and one ligand, dock the ligand into the prepared
    protein structure with OpenEye's Fred method:

     - modeling missing loops
     - building missing side chains
     - mutations, if `uniprot_id` or `sequence` attribute is provided for the protein component
       (see below)
     - removing everything but protein, water and ligand of interest
     - protonation at pH 7.4
     - perform Fred docking

    The protein component of each system must have a `pdb_id` or a `path` as well as a
    `pocket_resids` attribute specifying the structure to prepare and the binding pocket to dock
    to.

     - `pdb_id`: A string specifying the PDB entry of interest, required if `path` not given.
     - `path`: The path to the structure file, required if `pdb_id` not given.
     - `pocket_resids`: List of integers specifying the residues in the binding pocket of interest.

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

    The ligand component of each system must have a `to_smiles` method allowing access to the
    molecular structure to dock, e.g. a Ligand object from `core.ligands` initiated as
    Ligand.from_smiles("CCOCCC").

    Additionally, the ligand component can have the following optional attributes:

     - `name`: A string specifying the name of the ligand, will be used for generating the
       output file name.

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
    pKa_norm: bool, default=True
        Assign the predominant ionization state of the molecules to dock at pH ~7.4. If False,
        the ionization state of the input molecules will be conserved.
    """
    from MDAnalysis.core.universe import Universe

    def __init__(self, pKa_norm: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.pKa_norm = pKa_norm

    _SUPPORTED_TYPES = (ProteinLigandComplex,)

    def _featurize_one(self, system: ProteinLigandComplex) -> Union[Universe, None]:
        """
        Prepare a protein structure and dock a ligand using OpenEye's Fred method.

        Parameters
        ----------
        system: ProteinLigandComplex
            A system object holding a protein and a ligand component.

        Returns
        -------
        : Universe or None
            An MDAnalysis universe of the featurized system. None if no design unit or docking
            pose was found.
        """
        from pathlib import Path

        import MDAnalysis as mda
        from openeye import oechem, oedocking

        from ..docking.OEDocking import fred_docking, resids_to_box_molecule

        structure = self._read_protein_structure(system.protein)
        if structure is None:
            logger.warning(
                f"Could not read protein structure for {system.protein}, returning None!"
            )
            return None

        logger.debug("Preparing protein ligand complex ...")
        design_unit = self._get_design_unit(
            structure=structure,
            chain_id=system.protein.chain_id if hasattr(system.protein, "chain_id") else None,
            alternate_location=system.protein.alternate_location if hasattr(
                system.protein, "alternate_location"
            ) else None,
            has_ligand=hasattr(system.protein, "expo_id"),
            ligand_name=system.protein.expo_id if hasattr(system.protein, "expo_id") else None,
            model_loops_and_caps=False if system.protein.sequence else True,
        )  # if sequence is given model loops and caps separately later
        if not design_unit:
            logger.debug("No design unit found, returning None!")
            return None

        logger.debug("Extracting design unit components ...")
        protein, solvent = self._get_components(
            design_unit=design_unit,
            chain_id=system.protein.chain_id if hasattr(system.protein, "chain_id") else None
        )[:-1]

        logger.debug("Defining binding site ...")
        box_molecule = resids_to_box_molecule(protein, system.protein.pocket_resids)

        if system.protein.sequence:
            first_id = 1
            if "construct_range" in system.protein.metadata.keys():
                first_id = int(system.protein.metadata["construct_range"].split("-")[0])
            protein = self._process_protein(
                protein_structure=protein,
                amino_acid_sequence=system.protein.sequence,
                first_id=first_id
            )
            if not oechem.OEUpdateDesignUnit(  # does not work if no ligand was present
                    design_unit, protein, oechem.OEDesignUnitComponents_Protein
            ):
                # create new design unit with dummy site residue
                hierview = oechem.OEHierView(protein)
                first_residue = list(hierview.GetResidues())[0]
                design_unit = oechem.OEDesignUnit(
                    protein,
                    [f"{first_residue.GetResidueName()}:{first_residue.GetResidueNumber()}: :A"],
                    solvent
                )

        logger.debug("Preparing receptor for docking ...")
        receptor_options = oedocking.OEMakeReceptorOptions()
        receptor_options.SetBoxMol(box_molecule)
        oedocking.OEMakeReceptor(design_unit, receptor_options)

        logger.debug("Performing docking ...")
        docking_poses = fred_docking(
            design_unit,
            [system.ligand.molecule.to_openeye()],
            pKa_norm=self.pKa_norm
        )
        if not docking_poses:
            logger.debug("No docking pose found, returning None!")
            return None
        else:
            docking_pose = docking_poses[0]
        # generate residue information
        oechem.OEPerceiveResidues(docking_pose, oechem.OEPreserveResInfo_None)

        logger.debug("Assembling components ...")
        protein_ligand_complex = self._assemble_components(protein, solvent, docking_pose)

        logger.debug("Updating pdb header ...")
        protein_ligand_complex = self._update_pdb_header(
            protein_ligand_complex,
            protein_name=system.protein.name,
            ligand_name=system.ligand.name,
        )

        logger.debug("Writing results ...")
        file_path = self._write_results(
            protein_ligand_complex,
            "_".join([info for info in [
                system.protein.name,
                system.protein.pdb_id if system.protein.pdb_id
                else Path(system.protein.metadata["file_path"]).stem,
                f"chain{system.protein.chain_id}" if hasattr(system.protein, "chain_id")
                else None,
                f"altloc{system.protein.alternate_location}"
                if hasattr(system.protein, "alternate_location") else None,
            ] if info]),
            system.ligand.name,
        )

        logger.debug("Generating new MDAnalysis universe ...")
        structure = mda.Universe(file_path, in_memory=True)

        if not self.output_dir:
            logger.debug("Removing structure file ...")
            file_path.unlink()

        return structure


class OEHybridDockingFeaturizer(OEBaseModelingFeaturizer, SingleLigandProteinComplexFeaturizer):
    """
    Given systems with exactly one protein and one ligand, dock the ligand into the prepared
    protein structure with OpenEye's Hybrid method:

     - modeling missing loops
     - building missing side chains
     - mutations, if `uniprot_id` or `sequence` attribute is provided for the protein component
       (see below)
     - removing everything but protein, water and ligand of interest
     - protonation at pH 7.4
     - perform Hybrid docking

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
     - `expo_id`: A string specifying the ligand to use during Hybrid docking. This is especially
       useful if multiple ligands are present in a PDB structure.
     - `uniprot_id`: A string specifying the UniProt ID that will be used to fetch the amino acid
       sequence from UniProt, which will be used for modeling the protein. This will supersede the
       sequence information given in the PDB header.
     - `sequence`: An `AminoAcidSequence` object specifying the amino acid sequence that should be
       used during modeling the protein. This will supersede a given `uniprot_id` and the sequence
       information given in the PDB header.

    The ligand component of each system must have a `to_smiles` method allowing access to the
    molecular structure to dock, e.g. a Ligand object from `core.ligands` initiated as
    Ligand.from_smiles("CCOCCC").

    Additionally, the ligand component can have the following optional attributes:

     - `name`: A string specifying the name of the ligand, will be used for generating the
       output file name.

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
    pKa_norm: bool, default=True
        Assign the predominant ionization state of the molecules to dock at pH ~7.4. If False,
        the ionization state of the input molecules will be conserved.

    Note
    ----
    The provided protein must have a co-crystallized ligand to allow docking with OpenEye's
    Hybrid method.
    """
    from MDAnalysis.core.universe import Universe

    def __init__(self, pKa_norm: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.pKa_norm = pKa_norm

    _SUPPORTED_TYPES = (ProteinLigandComplex,)

    def _featurize_one(self, system: ProteinLigandComplex) -> Union[Universe, None]:
        """
        Prepare a protein structure and dock a ligand using OpenEye's Hybrid method.

        Parameters
        ----------
        system: ProteinLigandComplex
            A system object holding a protein and a ligand component.

        Returns
        -------
        : Universe or None
            An MDAnalysis universe of the featurized system. None if no design unit or docking
            pose was found.
        """
        from pathlib import Path

        import MDAnalysis as mda
        from openeye import oechem, oedocking

        from ..docking.OEDocking import hybrid_docking

        structure = self._read_protein_structure(system.protein)
        if structure is None:
            logger.warning(
                f"Could not read protein structure for {system.protein}, returning None!"
            )
            return None

        logger.debug("Preparing protein ligand complex ...")
        design_unit = self._get_design_unit(
            structure=structure,
            chain_id=system.protein.chain_id if hasattr(system.protein, "chain_id") else None,
            alternate_location=system.protein.alternate_location if hasattr(
                system.protein, "alternate_location"
            ) else None,
            has_ligand=True,
            ligand_name=system.protein.expo_id if hasattr(system.protein, "expo_id") else None,
            model_loops_and_caps=False if system.protein.sequence else True,
        )  # if sequence is given model loops and caps separately later
        if not design_unit:
            logger.debug("No design unit found, returning None!")
            return None

        logger.debug("Extracting design unit components ...")
        protein, solvent, ligand = self._get_components(
            design_unit=design_unit,
            chain_id=system.protein.chain_id if hasattr(system.protein, "chain_id") else None
        )

        if system.protein.sequence:
            first_id = 1
            if "construct_range" in system.protein.metadata.keys():
                first_id = int(system.protein.metadata["construct_range"].split("-")[0])
            protein = self._process_protein(
                protein_structure=protein,
                amino_acid_sequence=system.protein.sequence,
                first_id=first_id,
                ligand=ligand,
            )
            oechem.OEUpdateDesignUnit(design_unit, protein, oechem.OEDesignUnitComponents_Protein)

        if not design_unit.HasReceptor():
            logger.debug("Preparing receptor for docking ...")
            oedocking.OEMakeReceptor(design_unit)

        logger.debug("Performing docking ...")
        docking_poses = hybrid_docking(
            design_unit,
            [system.ligand.molecule.to_openeye()],
            pKa_norm=self.pKa_norm
        )
        if not docking_poses:
            logger.debug("No docking pose found, returning None!")
            return None
        else:
            docking_pose = docking_poses[0]
        # generate residue information
        oechem.OEPerceiveResidues(docking_pose, oechem.OEPreserveResInfo_None)

        logger.debug("Assembling components ...")
        protein_ligand_complex = self._assemble_components(protein, solvent, docking_pose)

        logger.debug("Updating pdb header ...")
        protein_ligand_complex = self._update_pdb_header(
            protein_ligand_complex,
            protein_name=system.protein.name,
            ligand_name=system.ligand.name,
        )

        logger.debug("Writing results ...")
        file_path = self._write_results(
            protein_ligand_complex,
            "_".join([info for info in [
                system.protein.name,
                system.protein.pdb_id if system.protein.pdb_id
                else Path(system.protein.metadata["file_path"]).stem,
                f"chain{system.protein.chain_id}" if hasattr(system.protein, "chain_id")
                else None,
                f"altloc{system.protein.alternate_location}"
                if hasattr(system.protein, "alternate_location") else None,
            ] if info]),
            system.ligand.name,
        )

        logger.debug("Generating new MDAnalysis universe ...")
        structure = mda.Universe(file_path, in_memory=True)

        if not self.output_dir:
            logger.debug("Removing structure file ...")
            file_path.unlink()

        return structure


class OEPositDockingFeaturizer(OEBaseModelingFeaturizer, SingleLigandProteinComplexFeaturizer):
    """
    Given systems with exactly one protein and one ligand, dock the ligand into the prepared
    protein structure with OpenEye's POSIT method:

     - modeling missing loops
     - building missing side chains
     - mutations, if `uniprot_id` or `sequence` attribute is provided for the protein component
       (see below)
     - removing everything but protein, water and ligand of interest
     - protonation at pH 7.4
     - perform Posit docking

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
     - `expo_id`: A string specifying the ligand to use during Posit docking. This is especially
       useful if multiple ligands are present in a PDB structure.
     - `uniprot_id`: A string specifying the UniProt ID that will be used to fetch the amino acid
       sequence from UniProt, which will be used for modeling the protein. This will supersede the
       sequence information given in the PDB header.
     - `sequence`: An `AminoAcidSequence` object specifying the amino acid sequence that should be
       used during modeling the protein. This will supersede a given `uniprot_id` and the sequence
       information given in the PDB header.

    The ligand component of each system must have a `to_smiles` method allowing access to the
    molecular structure to dock, e.g. a Ligand object from `core.ligands` initiated as
    Ligand.from_smiles("CCOCCC").

    Additionally, the ligand component can have the following optional attributes.

     - `name`: A string specifying the name of the ligand, will be used for generating the
       output file name.

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
    pKa_norm: bool, default=True
        Assign the predominant ionization state of the molecules to dock at pH ~7.4. If False,
        the ionization state of the input molecules will be conserved.

    Note
    ----
    The provided protein must have a co-crystallized ligand to allow docking with OpenEye's
    Posit method.
    """
    from MDAnalysis.core.universe import Universe

    def __init__(self, pKa_norm: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.pKa_norm = pKa_norm

    _SUPPORTED_TYPES = (ProteinLigandComplex,)

    def _featurize_one(self, system: ProteinLigandComplex) -> Union[Universe, None]:
        """
        Prepare a protein structure and dock a ligand using OpenEye's Posit method.

        Parameters
        ----------
        system: ProteinLigandComplex
            A system object holding a protein and a ligand component.

        Returns
        -------
        : Universe or None
            An MDAnalysis universe of the featurized system. None if no design unit, docking
            template ligand or docking pose was found.
        """
        from pathlib import Path

        import MDAnalysis as mda
        from openeye import oechem, oedocking

        from ..docking.OEDocking import pose_molecules

        structure = self._read_protein_structure(system.protein)
        if structure is None:
            logger.warning(
                f"Could not read protein structure for {system.protein}, returning None!"
            )
            return None

        logger.debug("Preparing protein ligand complex ...")
        design_unit = self._get_design_unit(
            structure=structure,
            chain_id=system.protein.chain_id if hasattr(system.protein, "chain_id") else None,
            alternate_location=system.protein.alternate_location if hasattr(
                system.protein, "alternate_location"
            ) else None,
            has_ligand=True,
            ligand_name=system.protein.expo_id if hasattr(system.protein, "expo_id") else None,
            model_loops_and_caps=False if system.protein.sequence else True,
        )  # if sequence is given model loops and caps separately later
        if not design_unit:
            logger.debug("No design unit found, returning None!")
            return None

        logger.debug("Extracting design unit components ...")
        protein, solvent, ligand = self._get_components(
            design_unit=design_unit,
            chain_id=system.protein.chain_id if hasattr(system.protein, "chain_id") else None
        )

        if system.protein.sequence:
            first_id = 1
            if "construct_range" in system.protein.metadata.keys():
                first_id = int(system.protein.metadata["construct_range"].split("-")[0])
            protein = self._process_protein(
                protein_structure=protein,
                amino_acid_sequence=system.protein.sequence,
                first_id=first_id,
                ligand=ligand,
            )
            oechem.OEUpdateDesignUnit(design_unit, protein, oechem.OEDesignUnitComponents_Protein)

        if not design_unit.HasReceptor():
            logger.debug("Preparing receptor for docking ...")
            oedocking.OEMakeReceptor(design_unit)

        logger.debug("Performing docking ...")
        docking_poses = pose_molecules(
            design_unit,
            [system.ligand.molecule.to_openeye()],
            pKa_norm=self.pKa_norm,
            score_pose=True,
        )
        if not docking_poses:
            logger.debug("No docking pose found, returning None!")
            return None
        else:
            docking_pose = docking_poses[0]
        # generate residue information
        oechem.OEPerceiveResidues(docking_pose, oechem.OEPreserveResInfo_None)

        logger.debug("Assembling components ...")
        protein_ligand_complex = self._assemble_components(protein, solvent, docking_pose)

        logger.debug("Updating pdb header ...")
        protein_ligand_complex = self._update_pdb_header(
            protein_ligand_complex,
            protein_name=system.protein.name,
            ligand_name=system.ligand.name,
        )

        logger.debug("Writing results ...")
        file_path = self._write_results(
            protein_ligand_complex,
            "_".join([info for info in [
                system.protein.name,
                system.protein.pdb_id if system.protein.pdb_id
                else Path(system.protein.metadata["file_path"]).stem,
                f"chain{system.protein.chain_id}" if hasattr(system.protein, "chain_id")
                else None,
                f"altloc{system.protein.alternate_location}"
                if hasattr(system.protein, "alternate_location") else None,
            ] if info]),
            system.ligand.name,
        )

        logger.debug("Generating new MDAnalysis universe ...")
        structure = mda.Universe(file_path, in_memory=True)

        if not self.output_dir:
            logger.debug("Removing structure file ...")
            file_path.unlink()

        return structure
