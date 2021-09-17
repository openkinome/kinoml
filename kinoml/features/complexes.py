"""
Featurizers that can only get applied to ProteinLigandComplexes or
subclasses thereof
"""
import logging

from .core import OEBaseModelingFeaturizer
from ..core.systems import ProteinLigandComplex


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
    from MDAnalysis.core import universe

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    _SUPPORTED_TYPES = (ProteinLigandComplex,)

    def _featurize_one(self, system: ProteinLigandComplex) -> universe:
        """
        Prepare a protein structure.

        Parameters
        ----------
        system: ProteinLigandComplex
            A system object holding a protein and a ligand component.

        Returns
        -------
        : universe
            An MDAnalysis universe of the featurized system.
        """
        import MDAnalysis as mda

        from ..modeling.OEModeling import read_molecules

        logging.debug("Interpreting system ...")
        system_dict = self._interpret_system(system)

        logging.debug("Reading structure ...")
        structure = read_molecules(system_dict["protein_path"])[0]

        logging.debug("Preparing protein ligand complex ...")
        design_unit = self._get_design_unit(
            structure=structure,
            chain_id=system_dict["protein_chain_id"],
            alternate_location=system_dict["protein_alternate_location"],
            has_ligand=True,
            ligand_name=system_dict["protein_expo_id"],
            model_loops_and_caps=False if system_dict["protein_sequence"] else True,
        )  # if sequence is given model loops and caps separately later

        logging.debug("Extracting design unit components ...")
        protein, solvent, ligand = self._get_components(
            design_unit, system_dict["protein_chain_id"]
        )

        if system_dict["protein_sequence"]:
            protein = self._process_protein(protein, system_dict["protein_sequence"])

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
                system_dict["protein_name"],
                system_dict["protein_pdb_id"] if system_dict["protein_pdb_id"]
                else system_dict["protein_path"].stem,
                f"chain{system_dict['protein_chain_id']}",
                f"altloc{system_dict['protein_alternate_location']}"
            ]),
            system_dict["ligand_name"],
        )

        logging.debug("Generating new MDAnalysis universe ...")
        structure = mda.Universe(file_path, in_memory=True)

        if not self.output_dir:
            logging.debug("Removing structure file ...")
            file_path.unlink()

        return structure


class OEHybridDockingFeaturizer(OEBaseModelingFeaturizer):
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
    from MDAnalysis.core import universe

    def __init__(self, pKa_norm: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.pKa_norm = pKa_norm

    _SUPPORTED_TYPES = (ProteinLigandComplex,)

    def _featurize_one(self, system: ProteinLigandComplex) -> universe:
        """
        Prepare a protein structure and dock a ligand using OpenEye's Hybrid method.

        Parameters
        ----------
        system: ProteinLigandComplex
            A system object holding a protein and a ligand component.

        Returns
        -------
        : universe
            An MDAnalysis universe of the featurized system.
        """
        import MDAnalysis as mda
        from openeye import oechem, oedocking

        from ..docking.OEDocking import hybrid_docking
        from ..modeling.OEModeling import read_smiles, read_molecules

        logging.debug("Interpreting system ...")
        system_dict = self._interpret_system(system)

        logging.debug("Reading structure ...")
        structure = read_molecules(system_dict["protein_path"])[0]

        logging.debug("Preparing protein ligand complex ...")
        design_unit = self._get_design_unit(
            structure=structure,
            chain_id=system_dict["protein_chain_id"],
            alternate_location=system_dict["protein_alternate_location"],
            has_ligand=True,
            ligand_name=system_dict["protein_expo_id"],
            model_loops_and_caps=False if system_dict["protein_sequence"] else True,
        )  # if sequence is given model loops and caps separately later

        logging.debug("Extracting design unit components ...")
        protein, solvent, ligand = self._get_components(
            design_unit, system_dict["protein_chain_id"]
        )

        if system_dict["protein_sequence"]:
            protein = self._process_protein(protein, system_dict["protein_sequence"])
            oechem.OEUpdateDesignUnit(design_unit, protein, oechem.OEDesignUnitComponents_Protein)

        if not design_unit.HasReceptor():
            logging.debug("Preparing receptor for docking ...")
            oedocking.OEMakeReceptor(design_unit)

        logging.debug("Performing docking ...")
        docking_pose = hybrid_docking(
            design_unit,
            [read_smiles(system.ligand.to_smiles())],
            pKa_norm=self.pKa_norm
        )[0]
        # generate residue information
        oechem.OEPerceiveResidues(docking_pose, oechem.OEPreserveResInfo_None)

        logging.debug("Assembling components ...")
        protein_ligand_complex = self._assemble_components(protein, solvent, docking_pose)

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
                system_dict["protein_name"],
                system_dict["protein_pdb_id"] if system_dict["protein_pdb_id"]
                else system_dict["protein_path"].stem,
                f"chain{system_dict['protein_chain_id']}",
                f"altloc{system_dict['protein_alternate_location']}"
            ]),
            system.ligand.name,
        )

        logging.debug("Generating new MDAnalysis universe ...")
        structure = mda.Universe(file_path, in_memory=True)

        if not self.output_dir:
            logging.debug("Removing structure file ...")
            file_path.unlink()

        return structure


class OEFredDockingFeaturizer(OEBaseModelingFeaturizer):
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
    from MDAnalysis.core import universe

    def __init__(self, pKa_norm: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.pKa_norm = pKa_norm

    _SUPPORTED_TYPES = (ProteinLigandComplex,)

    def _featurize_one(self, system: ProteinLigandComplex) -> universe:
        """
        Prepare a protein structure and dock a ligand using OpenEye's Fred method.

        Parameters
        ----------
        system: ProteinLigandComplex
            A system object holding a protein and a ligand component.

        Returns
        -------
        : universe
            An MDAnalysis universe of the featurized system.
        """
        import MDAnalysis as mda
        from openeye import oechem, oedocking

        from ..docking.OEDocking import fred_docking, resids_to_box_molecule
        from ..modeling.OEModeling import read_smiles, read_molecules

        logging.debug("Interpreting system ...")
        system_dict = self._interpret_system(system)

        logging.debug("Reading structure ...")
        structure = read_molecules(system_dict["protein_path"])[0]

        logging.debug("Preparing protein ligand complex ...")
        design_unit = self._get_design_unit(
            structure=structure,
            chain_id=system_dict["protein_chain_id"],
            alternate_location=system_dict["protein_alternate_location"],
            has_ligand=True if system_dict["protein_expo_id"] else False,
            ligand_name=system_dict["protein_expo_id"],
            model_loops_and_caps=False if system_dict["protein_sequence"] else True,
        )  # if sequence is given model loops and caps separately later

        logging.debug("Extracting design unit components ...")
        protein, solvent, ligand = self._get_components(
            design_unit, system_dict["protein_chain_id"]
        )

        logging.debug("Defining binding site ...")
        box_molecule = resids_to_box_molecule(protein, system.protein.pocket_resids)

        if system_dict["protein_sequence"]:
            protein = self._process_protein(protein, system_dict["protein_sequence"])
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

        logging.debug("Preparing receptor for docking ...")
        receptor_options = oedocking.OEMakeReceptorOptions()
        receptor_options.SetBoxMol(box_molecule)
        oedocking.OEMakeReceptor(design_unit, receptor_options)

        logging.debug("Performing docking ...")
        docking_pose = fred_docking(
            design_unit,
            [read_smiles(system.ligand.to_smiles())],
            pKa_norm=self.pKa_norm
        )[0]
        # generate residue information
        oechem.OEPerceiveResidues(docking_pose, oechem.OEPreserveResInfo_None)

        logging.debug("Assembling components ...")
        protein_ligand_complex = self._assemble_components(protein, solvent, docking_pose)

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
                system_dict["protein_name"],
                system_dict["protein_pdb_id"] if system_dict["protein_pdb_id"]
                else system_dict["protein_path"].stem,
                f"chain{system_dict['protein_chain_id']}",
                f"altloc{system_dict['protein_alternate_location']}"
            ]),
            system.ligand.name,
        )

        logging.debug("Generating new MDAnalysis universe ...")
        structure = mda.Universe(file_path, in_memory=True)

        if not self.output_dir:
            logging.debug("Removing structure file ...")
            file_path.unlink()

        return structure


class OEPositDockingFeaturizer(OEBaseModelingFeaturizer):
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
    Posit method.
    """
    from MDAnalysis.core import universe

    def __init__(self, pKa_norm: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.pKa_norm = pKa_norm

    _SUPPORTED_TYPES = (ProteinLigandComplex,)

    def _featurize_one(self, system: ProteinLigandComplex) -> universe:
        """
        Prepare a protein structure and dock a ligand using OpenEye's Posit method.

        Parameters
        ----------
        system: ProteinLigandComplex
            A system object holding a protein and a ligand component.

        Returns
        -------
        : universe
            An MDAnalysis universe of the featurized system.
        """
        import MDAnalysis as mda
        from openeye import oechem, oedocking

        from ..docking.OEDocking import pose_molecules
        from ..modeling.OEModeling import read_smiles, read_molecules

        logging.debug("Interpreting system ...")
        system_dict = self._interpret_system(system)

        logging.debug("Reading structure ...")
        structure = read_molecules(system_dict["protein_path"])[0]

        logging.debug("Preparing protein ligand complex ...")
        design_unit = self._get_design_unit(
            structure=structure,
            chain_id=system_dict["protein_chain_id"],
            alternate_location=system_dict["protein_alternate_location"],
            has_ligand=True,
            ligand_name=system_dict["protein_expo_id"],
            model_loops_and_caps=False if system_dict["protein_sequence"] else True,
        )  # if sequence is given model loops and caps separately later

        logging.debug("Extracting design unit components ...")
        protein, solvent, ligand = self._get_components(
            design_unit, system_dict["protein_chain_id"]
        )

        if system_dict["protein_sequence"]:
            protein = self._process_protein(protein, system_dict["protein_sequence"])
            oechem.OEUpdateDesignUnit(design_unit, protein, oechem.OEDesignUnitComponents_Protein)

        if not design_unit.HasReceptor():
            logging.debug("Preparing receptor for docking ...")
            oedocking.OEMakeReceptor(design_unit)

        logging.debug("Performing docking ...")
        docking_pose = pose_molecules(
            design_unit,
            [read_smiles(system.ligand.to_smiles())],
            pKa_norm=self.pKa_norm
        )[0]
        # generate residue information
        oechem.OEPerceiveResidues(docking_pose, oechem.OEPreserveResInfo_None)

        logging.debug("Assembling components ...")
        protein_ligand_complex = self._assemble_components(protein, solvent, docking_pose)

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
                system_dict["protein_name"],
                system_dict["protein_pdb_id"] if system_dict["protein_pdb_id"]
                else system_dict["protein_path"].stem,
                f"chain{system_dict['protein_chain_id']}",
                f"altloc{system_dict['protein_alternate_location']}"
            ]),
            system.ligand.name,
        )

        logging.debug("Generating new MDAnalysis universe ...")
        structure = mda.Universe(file_path, in_memory=True)

        if not self.output_dir:
            logging.debug("Removing structure file ...")
            file_path.unlink()

        return structure
