"""
Featurizers that can only get applied to ProteinLigandComplexes or
subclasses thereof
"""
import logging
from pathlib import Path
from typing import Iterable, Tuple, Union

from .core import ParallelBaseFeaturizer
from ..core.proteins import BaseProtein
from ..core.systems import System, ProteinSystem, ProteinLigandComplex


class OEBaseComplexFeaturizer(ParallelBaseFeaturizer):
    """
    This abstract class defines several methods that can be used by subclasses.

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
    from openeye import oechem

    def __init__(
            self,
            loop_db: Union[str, None] = None,
            cache_dir: Union[str, Path, None] = None,
            output_dir: Union[str, Path, None] = None,
            **kwargs,
    ):
        from appdirs import user_cache_dir

        super().__init__(**kwargs)
        self.loop_db = loop_db
        self.cache_dir = Path(user_cache_dir())
        self.output_dir = None
        if cache_dir:
            self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if output_dir:
            self.output_dir = Path(output_dir).expanduser().resolve()
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _read_protein_structure(self, protein: BaseProtein) -> oechem.OEGraphMol:
        """
        Interpret the given protein component and retrieve an OpenEye molecule holding the protein
        structure.

        Parameters
        ----------
        protein: BaseProtein
            The protein component.

        Returns
        -------
        structure: oechem.OEGraphMol
            An OpenEye molecule holding the protein structure.
        """
        from ..core.sequences import AminoAcidSequence
        from ..modeling.OEModeling import read_molecules
        from ..utils import FileDownloader, LocalFileStorage

        logging.debug("Checking for existing attributes ...")
        if not hasattr(protein, "pdb_id") and not hasattr(protein, "pdb_id"):
            raise AttributeError(
                f"The {self.__class__.__name__} requires systems with protein components having an"
                f" `pdb_id` or `path` attribute."
            )

        logging.debug("Interpreting protein structure ...")
        if hasattr(protein, "pdb_id"):
            protein.path = LocalFileStorage.rcsb_structure_pdb(protein.pdb_id, self.cache_dir)
            if protein.path.is_file():
                logging.debug(
                    f"Downloading protein structure {protein.pdb_id} from PDB ..."
                )
                FileDownloader.rcsb_structure_pdb(protein.pdb_id, self.cache_dir)
        logging.debug(f"Reading protein structure from {protein.path} ...")
        protein = read_molecules(protein.path)[0]

        logging.debug(f"Interpreting protein sequence ...")
        if not hasattr(protein, "sequence"):
            if hasattr(protein, "uniprot_id"):
                logging.debug(
                    f"Retrieving amino acid sequence details for UniProt entry "
                    f"{protein.uniprot_id} ..."
                )
                protein.sequence = AminoAcidSequence.from_uniprot(protein.uniprot_id)
        else:
            if not isinstance(protein.sequence, AminoAcidSequence):
                raise AttributeError(
                    f"The {self.__class__.__name__} only accepts systems with protein components whose"
                    f" `sequence` attribute is an instance of `AminoAcidSequence`."
                )

        return protein

    def _get_design_unit(self, system: Union[ProteinSystem, ProteinLigandComplex]) -> oechem.OEDesignUnit:
        """
        Get an OpenEye design unit from a system.

        Parameters
        ----------
        system: System
            A system with a protein component and optionally a ligand component.

        Returns
        -------
        design_unit: oechem.OEDesignUnit
            The design unit.
        """
        from openeye import oechem

        from ..modeling.OEModeling import prepare_complex, prepare_protein
        from ..utils import LocalFileStorage, sha256_objects

        structure = self._read_protein_structure(system.protein)

        design_unit_path = LocalFileStorage.featurizer_result(
            self.__class__.__name__,
            sha256_objects([self.loop_db, system]),
            "oedu",
            self.cache_dir,
        )
        if not design_unit_path.is_file():
            logging.debug("Generating design unit ...")
            if hasattr(system.protein, "sequence"):
                # model loops and caps later separately
                if hasattr(system, "ligand"):
                    design_unit = prepare_complex(
                        structure,
                        loop_db=None,
                        ligand_name=getattr(system.ligand, "expo_id", None),
                        chain_id=getattr(system.protein, "chain_id", None),
                        alternate_location=getattr(system.protein, "alternate_location", None),
                        cap_termini=False
                    )
                else:
                    design_unit = prepare_protein(
                        structure,
                        loop_db=None,
                        chain_id=getattr(system.protein, "chain_id", None),
                        alternate_location=getattr(system.protein, "alternate_location", None),
                        cap_termini=False,
                    )
            else:
                # model loops and caps with built in OESpruce capabilities
                if hasattr(system, "ligand"):
                    design_unit = prepare_complex(
                        structure,
                        loop_db=self.loop_db,
                        ligand_name=getattr(system.ligand, "expo_id", None),
                        chain_id=getattr(system.protein, "chain_id", None),
                        alternate_location=getattr(system.protein, "alternate_location", None),
                        cap_termini=True
                    )
                else:
                    design_unit = prepare_protein(
                        structure,
                        loop_db=self.loop_db,
                        chain_id=getattr(system.protein, "chain_id", None),
                        alternate_location=getattr(system.protein, "alternate_location", None),
                        cap_termini=True
                    )
            logging.debug("Writing design unit ...")
            oechem.OEWriteDesignUnit(str(design_unit_path), design_unit)
        # re-reading design unit helps proper capping of e.g. 2itz
        # TODO: revisit, report bug
        logging.debug("Reading design unit from file ...")
        design_unit = oechem.OEDesignUnit()
        oechem.OEReadDesignUnit(str(design_unit_path), design_unit)

        return design_unit

    @staticmethod
    def _get_components(
        design_unit: oechem.OEDesignUnit
    ) -> Tuple[oechem.OEGraphMol(), oechem.OEGraphMol(), oechem.OEGraphMol()]:
        """
        Get protein, solvent and ligand components from an OpenEye design unit.
        Parameters
        ----------
        design_unit: oechem.OEDesignUnit
            The OpenEye design unit to extract components from.
        Returns
        -------
        components: tuple of oechem.OEGraphMol, oechem.OEGraphMol and oechem.OEGraphMol
            OpenEye molecules holding protein, solvent and ligand.
        """
        from openeye import oechem

        protein, solvent, ligand = oechem.OEGraphMol(), oechem.OEGraphMol(), oechem.OEGraphMol()

        logging.debug("Extracting molecular components ...")
        design_unit.GetProtein(protein)
        design_unit.GetSolvent(solvent)
        design_unit.GetLigand(ligand)

        # delete protein atoms with no name (found in prepared protein of 4ll0)
        for atom in protein.GetAtoms():
            if not atom.GetName().strip():
                logging.debug("Deleting unknown atom ...")
                protein.DeleteAtom(atom)

        # perceive residues to remove artifacts of other design units in the sequence of the protein
        # preserve certain properties to assure correct behavior of the pipeline,
        # e.g. deletion of chains in OEKLIFSKinaseApoFeaturizer._process_kinase_domain method
        preserved_info = (
                oechem.OEPreserveResInfo_ResidueNumber
                | oechem.OEPreserveResInfo_ResidueName
                | oechem.OEPreserveResInfo_AtomName
                | oechem.OEPreserveResInfo_ChainID
                | oechem.OEPreserveResInfo_HetAtom
                | oechem.OEPreserveResInfo_InsertCode
                | oechem.OEPreserveResInfo_AlternateLocation
        )
        oechem.OEPerceiveResidues(protein, preserved_info)
        oechem.OEPerceiveResidues(solvent, preserved_info)
        oechem.OEPerceiveResidues(ligand)

        logging.debug(
            "Number of component atoms: " +
            f"Protein - {protein.NumAtoms()}, " +
            f"Solvent - {solvent.NumAtoms()}, " +
            f"Ligand - {ligand.NumAtoms()}."
        )
        return protein, solvent, ligand

    def _assemble_components(
        self,
        protein: oechem.OEMolBase,
        solvent: oechem.OEMolBase,
        ligand: Union[oechem.OEMolBase, None]
    ) -> oechem.OEMolBase:
        """
        Assemble components of a solvated protein-ligand complex into a single OpenEye molecule.
        Parameters
        ----------
        protein: oechem.OEMolBase
            An OpenEye molecule holding the protein of interest.
        solvent: oechem.OEMolBase
            An OpenEye molecule holding the solvent of interest.
        ligand: oechem.OEMolBase or None
            An OpenEye molecule holding the ligand of interest if given.
        Returns
        -------
        assembled_components: oechem.OEMolBase
            An OpenEye molecule holding protein, solvent and ligand if given.
        """
        from openeye import oechem

        from ..modeling.OEModeling import update_residue_identifiers

        assembled_components = oechem.OEGraphMol()

        logging.debug("Adding protein ...")
        oechem.OEAddMols(assembled_components, protein)

        if ligand:
            logging.debug("Renaming ligand ...")
            for atom in ligand.GetAtoms():
                oeresidue = oechem.OEAtomGetResidue(atom)
                oeresidue.SetName("LIG")
                oechem.OEAtomSetResidue(atom, oeresidue)

            logging.debug("Adding ligand ...")
            oechem.OEAddMols(assembled_components, ligand)

        logging.debug("Adding water molecules ...")
        filtered_solvent = self._remove_clashing_water(solvent, ligand, protein)
        oechem.OEAddMols(assembled_components, filtered_solvent)

        logging.debug("Updating hydrogen positions of assembled components ...")
        options = oechem.OEPlaceHydrogensOptions()  # keep protonation state from docking
        predicate = oechem.OEAtomMatchResidue(["LIG:.*:.*:.*:.*"])
        options.SetBypassPredicate(predicate)
        oechem.OEPlaceHydrogens(assembled_components, options)
        # keep tyrosine protonated, e.g. 6tg1 chain B
        predicate = oechem.OEAndAtom(
            oechem.OEAtomMatchResidue(["TYR:.*:.*:.*:.*"]),
            oechem.OEHasFormalCharge(-1)
        )
        for atom in assembled_components.GetAtoms(predicate):
            if atom.GetName().strip() == "OH":
                atom.SetFormalCharge(0)
                atom.SetImplicitHCount(1)
        oechem.OEAddExplicitHydrogens(assembled_components)

        logging.debug("Updating residue identifiers ...")
        assembled_components = update_residue_identifiers(assembled_components)

        return assembled_components

    @staticmethod
    def _remove_clashing_water(
        solvent: oechem.OEMolBase,
        ligand: Union[oechem.OEMolBase, None],
        protein: oechem.OEMolBase
    ) -> oechem.OEGraphMol:
        """
        Remove water molecules clashing with a ligand or newly modeled protein residues.
        Parameters
        ----------
        solvent: oechem.OEGraphMol
            An OpenEye molecule holding the water molecules.
        ligand: oechem.OEGraphMol or None
            An OpenEye molecule holding the ligand or None.
        protein: oechem.OEGraphMol
            An OpenEye molecule holding the protein.
        Returns
        -------
         : oechem.OEGraphMol
            An OpenEye molecule holding water molecules not clashing with the ligand or newly modeled protein residues.
        """
        from openeye import oechem, oespruce
        from scipy.spatial import cKDTree

        from ..modeling.OEModeling import get_atom_coordinates, split_molecule_components

        if ligand is not None:
            ligand_heavy_atoms = oechem.OEGraphMol()
            oechem.OESubsetMol(
                ligand_heavy_atoms,
                ligand,
                oechem.OEIsHeavy()
            )
            ligand_heavy_atom_coordinates = get_atom_coordinates(ligand_heavy_atoms)
            ligand_heavy_atoms_tree = cKDTree(ligand_heavy_atom_coordinates)

        modeled_heavy_atoms = oechem.OEGraphMol()
        oechem.OESubsetMol(
            modeled_heavy_atoms,
            protein,
            oechem.OEAndAtom(
                oespruce.OEIsModeledAtom(),
                oechem.OEIsHeavy()
            )
        )
        modeled_heavy_atoms_tree = None
        if modeled_heavy_atoms.NumAtoms() > 0:
            modeled_heavy_atom_coordinates = get_atom_coordinates(modeled_heavy_atoms)
            modeled_heavy_atoms_tree = cKDTree(modeled_heavy_atom_coordinates)

        filtered_solvent = oechem.OEGraphMol()
        waters = split_molecule_components(solvent)
        # iterate over water molecules and check for clashes and ambiguous water molecules
        for water in waters:
            try:
                water_oxygen_atom = water.GetAtoms(oechem.OEIsOxygen()).next()
            except StopIteration:
                # experienced lonely water hydrogens for 2v7a after mutating PTR393 to TYR
                logging.debug("Removing water molecule without oxygen!")
                continue
            # experienced problems when preparing 4pmp
            # making design units generated clashing waters that were not protonatable
            # TODO: revisit this behavior
            if oechem.OEAtomGetResidue(water_oxygen_atom).GetInsertCode() != " ":
                logging.debug("Removing ambiguous water molecule!")
                continue
            water_oxygen_coordinates = water.GetCoords()[water_oxygen_atom.GetIdx()]
            # check for clashes with newly placed ligand
            if ligand is not None:
                clashes = ligand_heavy_atoms_tree.query_ball_point(water_oxygen_coordinates, 1.5)
                if len(clashes) > 0:
                    logging.debug("Removing water molecule clashing with ligand atoms!")
                    continue
            # check for clashes with newly modeled protein residues
            if modeled_heavy_atoms_tree:
                clashes = modeled_heavy_atoms_tree.query_ball_point(water_oxygen_coordinates, 1.5)
                if len(clashes) > 0:
                    logging.debug("Removing water molecule clashing with modeled atoms!")
                    continue
            # water molecule is not clashy, add to filtered solvent
            oechem.OEAddMols(filtered_solvent, water)

        return filtered_solvent

    def _update_pdb_header(
        self,
        structure: oechem.OEMolBase,
        protein_name: str,
        ligand_name: [str, None],
        other_pdb_header_info: Union[None, Iterable[Tuple[str, str]]] = None
    ) -> oechem.OEMolBase:
        """
        Stores information about Featurizer, protein and ligand in the PDB header COMPND section in the
        given OpenEye molecule.
        Parameters
        ----------
        structure: oechem.OEMolBase
            An OpenEye molecule.
        protein_name: str
            The name of the protein.
        ligand_name: str or None
            The name of the ligand if present.
        other_pdb_header_info: None or iterable of tuple of str
            Tuples with information that should be saved in the PDB header. Each tuple consists of two strings,
            i.e., the PDB header section (e.g. COMPND) and the respective information.
        Returns
        -------
        : oechem.OEMolBase
            The OpenEye molecule containing the updated PDB header.
        """
        from openeye import oechem

        oechem.OEClearPDBData(structure)
        oechem.OESetPDBData(structure, "COMPND", f"\tFeaturizer: {self.__class__.__name__}")
        oechem.OEAddPDBData(structure, "COMPND", f"\tProtein: {protein_name}")
        if ligand_name:
            oechem.OEAddPDBData(structure, "COMPND", f"\tLigand: {ligand_name}")
        if other_pdb_header_info is not None:
            for section, information in other_pdb_header_info:
                oechem.OEAddPDBData(structure, section, information)

        return structure

    def _write_results(
        self,
        structure: oechem.OEMolBase,
        protein_name: str,
        ligand_name: Union[str, None],
     ) -> Path:
        """
        Write the results from the Featurizer and retrieve the paths to protein or complex if a
        ligand is present.
        Parameters
        ----------
        structure: oechem.OEMolBase
            The OpenEye molecule holding the featurized system.
        protein_name: str
            The name of the protein.
        ligand_name: str or None
            The name of the ligand if present.
        Returns
        -------
        : Path
            Path to prepared protein or complex if ligand structure is present.
        """
        from openeye import oechem

        from ..modeling.OEModeling import write_molecules, remove_non_protein
        from ..utils import LocalFileStorage

        if self.output_dir:
            if ligand_name:
                logging.debug("Writing protein ligand complex ...")
                complex_path = LocalFileStorage.featurizer_result(
                    self.__class__.__name__,
                    f"{protein_name}_{ligand_name}_complex",
                    "oeb",
                    self.output_dir,
                )
                write_molecules([structure], complex_path)

                complex_path = LocalFileStorage.featurizer_result(
                    self.__class__.__name__,
                    f"{protein_name}_{ligand_name}_complex",
                    "pdb",
                    self.output_dir,
                )
                write_molecules([structure], complex_path)

                logging.debug("Splitting components")
                solvated_protein = remove_non_protein(structure, remove_water=False)
                split_options = oechem.OESplitMolComplexOptions()
                ligand = list(oechem.OEGetMolComplexComponents(
                    structure, split_options, split_options.GetLigandFilter())
                )[0]

                logging.debug("Writing protein ...")
                protein_path = LocalFileStorage.featurizer_result(
                    self.__class__.__name__,
                    f"{protein_name}_{ligand_name}_protein",
                    "oeb",
                    self.output_dir,
                )
                write_molecules([solvated_protein], protein_path)

                protein_path = LocalFileStorage.featurizer_result(
                    self.__class__.__name__,
                    f"{protein_name}_{ligand_name}_protein",
                    "pdb",
                    self.output_dir,
                )
                write_molecules([solvated_protein], protein_path)

                logging.debug("Writing ligand ...")
                ligand_path = LocalFileStorage.featurizer_result(
                    self.__class__.__name__,
                    f"{protein_name}_{ligand_name}_ligand",
                    "sdf",
                    self.output_dir,
                )
                write_molecules([ligand], ligand_path)

                return complex_path
            else:
                logging.debug("Writing protein ...")
                protein_path = LocalFileStorage.featurizer_result(
                    self.__class__.__name__,
                    f"{protein_name}_protein",
                    "oeb",
                    self.output_dir,
                )
                write_molecules([structure], protein_path)

                protein_path = LocalFileStorage.featurizer_result(
                    self.__class__.__name__,
                    f"{protein_name}_protein",
                    "pdb",
                    self.output_dir,
                )
                write_molecules([structure], protein_path)

                return protein_path
        else:
            if ligand_name:
                complex_path = LocalFileStorage.featurizer_result(
                    self.__class__.__name__,
                    f"{protein_name}_{ligand_name}_complex",
                    "pdb",
                )
                write_molecules([structure], complex_path)

                return complex_path
            else:
                protein_path = LocalFileStorage.featurizer_result(
                    self.__class__.__name__,
                    f"{protein_name}_protein",
                    "pdb",
                )
                write_molecules([structure], protein_path)

                return protein_path


class OEComplexFeaturizer(OEBaseComplexFeaturizer):
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
        from openeye import oechem

        logging.debug("Preparing complex structure ...")
        design_unit = self._get_design_unit(system)
