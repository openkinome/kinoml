"""
Featurizers that can only get applied to ProteinLigandComplexes or
subclasses thereof
"""
from __future__ import annotations
from functools import lru_cache
import logging
from typing import Union, Tuple, Iterable, List

from .core import BaseFeaturizer
from ..core.ligands import FileLigand, SmilesLigand
from ..core.proteins import FileProtein, PDBProtein
from ..core.sequences import Biosequence
from ..core.systems import Protein, ProteinLigandComplex


class OEHybridDockingFeaturizer(BaseFeaturizer):

    """
    Given a System with exactly one protein and one ligand,
    dock the ligand in the designated binding pocket.

    We assume that a smiles and file-based System object will be passed;
    this means we will have a System.components with FileProtein and
    FileLigand or SmilesLigand. The file itself could be a URL.

    Parameters
    ----------
    loop_db: str
        The path to the loop database used by OESpruce to model missing loops.
    """
    from openeye import oechem, oegrid

    def __init__(self, loop_db: Union[str, None] = None):
        self.loop_db = loop_db

    _SUPPORTED_TYPES = (ProteinLigandComplex,)

    @lru_cache(maxsize=100)
    def _featurize(self, system: ProteinLigandComplex) -> ProteinLigandComplex:
        """
        Perform hybrid docking with the OpenEye toolkit and thoughtful defaults.
        Parameters
        ----------
        system: ProteinLigandComplex
            A system object holding protein and ligand information.
        Returns
        -------
        protein_ligand_complex: ProteinLigandComplex
            The same system but with docked ligand.
        """
        from openeye import oechem

        from ..docking.OEDocking import create_hybrid_receptor, hybrid_docking

        logging.debug("Interpreting system ...")
        ligand, protein, electron_density = self._interpret_system(system)

        logging.debug("Preparing protein ligand complex ...")
        design_unit = self._get_design_unit(protein, system.protein.name, electron_density)

        logging.debug("Extracting components ...")
        prepared_protein, prepared_solvent, prepared_ligand = self._get_components(design_unit)  # TODO: rename prepared_ligand

        logging.debug("Creating hybrid receptor ...")
        hybrid_receptor = create_hybrid_receptor(prepared_protein, prepared_ligand)  # TODO: takes quite long, should save this somehow

        logging.debug("Performing docking ...")
        docking_pose = hybrid_docking(hybrid_receptor, [ligand])[0]
        oechem.OEPerceiveResidues(docking_pose, oechem.OEPreserveResInfo_None)  # generate residue information

        logging.debug("Assembling components ...")
        protein_ligand_complex = self._assemble_components(prepared_protein, prepared_solvent, docking_pose)

        logging.debug("Updating pdb header ...")
        protein_ligand_complex = self._update_pdb_header(
            protein_ligand_complex,
            protein_name=system.protein.name,
            ligand_name=system.ligand.name,
        )

        logging.debug("Writing results ...")
        protein_path, ligand_path = self._write_results(
            protein_ligand_complex,
            system.protein.name,
            system.ligand.name
        )

        logging.debug("Generating new system components ...")
        file_protein = FileProtein(path=protein_path)
        file_ligand = FileLigand(path=ligand_path)
        protein_ligand_complex = ProteinLigandComplex(
            components=[file_protein, file_ligand]
        )

        return protein_ligand_complex

    @staticmethod
    def _interpret_system(
        system: Union[Protein, ProteinLigandComplex]
    ) -> Tuple[Union[oechem.OEGraphMol, None], Union[oechem.OEGraphMol, None], Union[oegrid.OESkewGrid, None]]:
        """
        Interpret the given system components and retrieve OpenEye objects holding ligand, protein and electron density.
        Parameters
        ----------
        system: Protein or ProteinLigandComplex
            The system to featurize.
        Returns
        -------
        : tuple of oechem.OEGraphMol or None, oechem.OEGraphMol or None, and oegrid.OESkewGrid or None
            OpenEye objects holding ligand, protein and electron density
        """
        from ..modeling.OEModeling import (
            read_smiles,
            read_molecules,
            read_electron_density,
        )
        from ..utils import FileDownloader

        ligand = None
        if hasattr(system, "ligand"):
            logging.debug("Interpreting ligand ...")
            if isinstance(system.ligand, SmilesLigand):
                logging.debug("Loading ligand from SMILES string ...")
                ligand = read_smiles(system.ligand.smiles)
            elif isinstance(system.ligand, FileLigand):
                logging.debug(f"Loading ligand from {system.ligand.path} ...")
                ligand = read_molecules(system.ligand.path)[0]
            else:
                raise NotImplementedError("Provide SmilesLigand or FileLigand.")

        logging.debug("Interpreting protein ...")
        if hasattr(system.protein, "pdb_id"):
            if not system.protein.path.is_file():
                logging.debug(
                    f"Downloading protein structure {system.protein.pdb_id} from PDB ..."
                )
                FileDownloader.rcsb_structure_pdb(system.protein.pdb_id)
        logging.debug(f"Reading protein structure from {system.protein.path} ...")
        protein = read_molecules(system.protein.path)[0]

        logging.debug("Interpreting electron density ...")
        electron_density = None
        # if system.protein.electron_density_path is not None:
        #    if hasattr(system.protein, 'pdb_id'):
        #        if not system.protein.electron_density_path.is_file():
        #            logging.debug(f"Downloading electron density for structure {system.protein.pdb_id} from PDB ...")
        #            FileDownloader.rcsb_electron_density_mtz(system.protein.pdb_id)
        #    logging.debug(f"Reading electron density from {system.protein.electron_density_path} ...")
        # TODO: Kills Kernel for some reason
        #    electron_density = read_electron_density(system.protein.electron_density_path)

        return ligand, protein, electron_density

    def _get_design_unit(
        self,
        complex_structure: oechem.OEGraphMol,
        design_unit_identifier: str,
        electron_density: Union[oegrid.OESkewGrid, None]
    ) -> oechem.OEDesignUnit:
        """
        Get an OpenEye design unit from a protein ligand complex.
        Parameters
        ----------
        complex_structure: oechem.OEGraphMol
            An OpenEye molecule holding the protein in complex with a ligand.
        design_unit_identifier: str
            A unique identifier describing the design unit.
        electron_density: oegrid.OESkewGrid or None
            An OpenEye grid holding the electron density of the protein ligand complex.
        Returns
        -------
        : oechem.OEDesignUnit
            The design unit.
        """
        from openeye import oechem

        from ..modeling.OEModeling import prepare_complex
        from ..utils import LocalFileStorage

        design_unit_path = LocalFileStorage.featurizer_result(
            self.__class__.__name__,
            f"{design_unit_identifier}_design_unit", "oedu"
        )  # TODO: the file name needs to be unique
        if not design_unit_path.is_file():
            logging.debug("Generating design unit ...")
            design_unit = prepare_complex(complex_structure, electron_density, self.loop_db)
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
        protein: oechem.OEGraphMol,
        solvent: oechem.OEGraphMol,
        ligand: Union[oechem.OEGraphMol, None]
    ) -> oechem.OEGraphMol:
        """
        Assemble components of a solvated protein-ligand complex into a single OpenEye molecule.
        Parameters
        ----------
        protein: oechem.OEGraphMol
            An OpenEye molecule holding the protein of interest.
        solvent: oechem.OEGraphMol
            An OpenEye molecule holding the solvent of interest.
        ligand: oechem.OEGraphMol or None
            An OpenEye molecule holding the ligand of interest if given.
        Returns
        -------
        assembled_components: oechem.OEGraphMol
            An OpenEye molecule holding protein, solvent and ligand if given.
        """
        from openeye import oechem

        from ..modeling.OEModeling import update_residue_identifiers

        assembled_components = oechem.OEGraphMol()

        logging.debug("Adding protein ...")
        oechem.OEAddMols(assembled_components, protein)

        if ligand:
            logging.debug("Adding ligand ...")
            oechem.OEAddMols(assembled_components, ligand)

        logging.debug("Adding water molecules ...")
        filtered_solvent = self._remove_clashing_water(solvent, ligand, protein)
        oechem.OEAddMols(assembled_components, filtered_solvent)

        logging.debug("Updating hydrogen positions of assembled components ...")
        oechem.OEPlaceHydrogens(assembled_components)
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
        solvent: oechem.OEGraphMol,
        ligand: Union[oechem.OEGraphMol, None],
        protein: oechem.OEGraphMol
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
        # iterate over water molecules and check for clashes
        for water in waters:
            water_oxygen_atom = water.GetAtoms(oechem.OEIsOxygen()).next()
            # experienced problems when preparing 4pmp
            # making design units generated clashing waters that were not protonatable
            # TODO: revisit this behavior
            if oechem.OEAtomGetResidue(water_oxygen_atom).GetInsertCode() != " ":
                logging.debug("Found ambiguous water molecule!")
                continue
            water_oxygen_coordinates = water.GetCoords()[water_oxygen_atom.GetIdx()]
            # check for clashes with newly placed ligand
            if ligand is not None:
                clashes = ligand_heavy_atoms_tree.query_ball_point(water_oxygen_coordinates, 1.5)
                if len(clashes) > 0:
                    logging.debug("Found water molecule clashing with ligand atoms!")
                    continue
            # check for clashes with newly modeled protein residues
            if modeled_heavy_atoms_tree:
                clashes = modeled_heavy_atoms_tree.query_ball_point(water_oxygen_coordinates, 1.5)
                if len(clashes) > 0:
                    logging.debug("Found water molecule clashing with modeled atoms!")
                    continue
            # water molecule is not clashy, add to filtered solvent
            oechem.OEAddMols(filtered_solvent, water)

        return filtered_solvent

    def _update_pdb_header(
        self,
        structure: oechem.OEGraphMol,
        protein_name: str,
        ligand_name: [str, None],
        other_pdb_header_info: Union[None, Iterable[Tuple[str, str]]] = None
    ) -> oechem.OEGraphMol:
        """
        Stores information about Featurizer, protein and ligand in the PDB header COMPND section in the
        given OpenEye molecule.
        Parameters
        ----------
        structure: oechem.OEGraphMol
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
        : oechem.OEGraphMol
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
        structure: oechem.OEGraphMol,
        protein_name: str,
        ligand_name: Union[str, None]
     ) -> Tuple[str, Union[str, None]]:
        """
        Write the results from the Featurizer and retrieve the paths to protein and ligand if present.
        Parameters
        ----------
        structure: oechem.OEGraphMol
            The OpenEye molecule holding the featurized system.
        protein_name: str
            The name of the protein.
        ligand_name: str or None
            The name of the ligand if present.
        Returns
        -------
        : tuple of str and str or None
            Paths to prepared protein and docked ligand structure if present.
        """
        from openeye import oechem

        from ..modeling.OEModeling import write_molecules, remove_non_protein
        from ..utils import LocalFileStorage

        if ligand_name:
            logging.debug("Writing protein ligand complex ...")
            complex_path = LocalFileStorage.featurizer_result(
                self.__class__.__name__,
                f"{protein_name}_{ligand_name}_complex",
                "pdb")
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
                "pdb")
            write_molecules([solvated_protein], protein_path)

            logging.debug("Writing ligand ...")
            ligand_path = LocalFileStorage.featurizer_result(
                self.__class__.__name__,
                f"{protein_name}_{ligand_name}_ligand",
                "sdf")
            write_molecules([ligand], ligand_path)
        else:
            logging.debug("Writing protein ...")
            protein_path = LocalFileStorage.featurizer_result(
                self.__class__.__name__,
                f"{protein_name}_protein",
                "pdb")
            write_molecules([structure], protein_path)

            ligand_path = None

        return protein_path, ligand_path


class OEKLIFSKinaseApoFeaturizer(OEHybridDockingFeaturizer):
    """
    Given a System with exactly one kinase prepare an apo kinase.

    Parameters
    ----------
    loop_db: str or None
        The path to the loop database used by OESpruce to model missing loops.
    """
    from openeye import oechem, oegrid

    def __init__(self, loop_db: Union[str, None] = None):
        super().__init__(loop_db)
        self.loop_db = loop_db

    _SUPPORTED_TYPES = (Protein,)

    @lru_cache(maxsize=100)
    def _featurize(self, system: Protein) -> Protein:
        """
        Prepare a kinase using the OpenEye toolkit, the KLIFS database and thoughtful defaults.
        Parameters
        ----------
        system: Protein
            A system object holding protein information.
        Returns
        -------
        protein: Protein
            The same system but prepared.
        """
        from ..modeling.OEModeling import get_expression_tags, delete_residue
        from ..utils import LocalFileStorage

        logging.debug("Interpreting kinase of interest ...")
        self._interpret_kinase(system.protein)

        # select structure
        if hasattr(system.protein, "pdb_id"):
            kinase_details = self._select_kinase_structure_by_pdb_id(
                system.protein.pdb_id,
                system.protein.klifs_kinase_id,
                system.protein.chain_id,
                system.protein.alternate_location
            )
        else:
            kinase_details = self._select_kinase_structure_by_klifs_kinase_id(
                system.protein.klifs_kinase_id,
                system.protein.dfg,
                system.protein.ac_helix
            )

        if not all([
            hasattr(system.protein, "pdb_id"),
            hasattr(system.protein, "path"),
            hasattr(system.protein, "electron_density_path")
        ]):
            logging.debug(f"Adding attributes to BaseProtein ...")  # TODO: bad idea in a library
            system.protein.pdb_id = kinase_details["structure.pdb_id"]
            system.protein.path = LocalFileStorage.rcsb_structure_pdb(kinase_details["structure.pdb_id"])
            system.protein.electron_density_path = LocalFileStorage.rcsb_electron_density_mtz(
                kinase_details["structure.pdb_id"]
            )

        logging.debug("Interpreting system ...")
        kinase_structure, electron_density = self._interpret_system(system)[1:]

        logging.debug(f"Preparing kinase template structure of {kinase_details['structure.pdb_id']} ...")
        design_unit = self._get_design_unit(
            kinase_structure,
            structure_identifier=kinase_details["structure.pdb_id"],
            electron_density=electron_density,
            ligand_name=kinase_details["ligand.expo_id"],
            chain_id=kinase_details["structure.chain"],
            alternate_location=system.protein.alternate_location  # KLIFS alternate locations buggy, e.g. 3cs9
        )

        logging.debug("Extracting kinase and solvent from design unit ...")
        prepared_kinase, prepared_solvent = self._get_components(design_unit)[:-1]

        logging.debug("Deleting expression tags ...")
        expression_tags = get_expression_tags(kinase_structure)
        for expression_tag in expression_tags:
            prepared_kinase = delete_residue(
                prepared_kinase,
                chain_id=expression_tag["chain_id"],
                residue_name=expression_tag["residue_name"],
                residue_id=expression_tag["residue_id"]
            )

        logging.debug("Processing kinase domain ...")
        processed_kinase_domain = self._process_kinase_domain(
            prepared_kinase,
            system.protein.sequence,
            kinase_details["structure.chain"]
        )

        logging.debug("Assembling components ...")
        solvated_kinase = self._assemble_components(processed_kinase_domain, prepared_solvent, None)

        logging.debug("Updating pdb header ...")
        solvated_kinase = self._update_pdb_header(
            solvated_kinase,
            kinase_details["kinase.klifs_name"],
            None,
            [("COMPND", f"\tKinase template: {kinase_details['structure.pdb_id']}")]
        )

        logging.debug("Writing results ...")
        protein_path = self._write_results(
            solvated_kinase,
            "_".join([
                f"{kinase_details['kinase.klifs_name']}",
                f"{kinase_details['structure.pdb_id']}",
                f"chain{kinase_details['structure.chain']}",
                f"altloc{system.protein.alternate_location}"  # KLIFS alternate locations buggy, e.g. 3cs9
            ]),
            None
        )[0]

        logging.debug("Generating new system components")
        file_protein = FileProtein(path=protein_path)
        protein = Protein(components=[file_protein])

        return protein

    def _interpret_kinase(self, protein: Protein):
        """
        Interpret the kinase information stored in the given Protein object.
        Parameters
        ----------
        protein: Protein
            The Protein object.
        """
        from opencadd.databases.klifs import setup_remote

        from ..core.sequences import AminoAcidSequence

        remote = setup_remote()

        # identify kinase of interest and get KLIFS kinase ID and UniProt ID if not provided
        if any([
            hasattr(protein, "klifs_kinase_id"),
            hasattr(protein, "uniprot_id"),
            hasattr(protein, "pdb_id")
        ]):
            # add chain_id and alternate_location attributes if not present
            if not hasattr(protein, "chain_id"):
                protein.chain_id = None
            if not hasattr(protein, "alternate_location"):
                protein.alternate_location = None
            if protein.alternate_location == "-":
                protein.alternate_location = None
            # if pdb id is given, query KLIFS by pdb
            if hasattr(protein, "pdb_id"):
                structures = remote.structures.by_structure_pdb_id(
                    protein.pdb_id,
                    protein.alternate_location,
                    protein.chain_id
                )
                protein.klifs_kinase_id = structures["kinase.klifs_id"].iloc[0]
            # if KLIFS kinase ID is not given, query by UniProt ID
            if not hasattr(protein, "klifs_kinase_id"):
                logging.debug("Converting UniProt ID to KLIFS kinase ID ...")
                kinase_ids = remote.kinases.all_kinases()["kinase.klifs_id"].to_list()
                kinases = remote.kinases.by_kinase_klifs_id(kinase_ids)
                protein.klifs_kinase_id = kinases[
                    kinases["kinase.uniprot"] == protein.uniprot_id]["kinase.klifs_id"].iloc[0]
            # if UniProt ID is not given, query by KLIFS kinase ID
            if not hasattr(protein, "uniprot_id"):
                logging.debug("Converting KLIFS kinase ID to UniProt ID  ...")
                kinase_ids = remote.kinases.all_kinases()["kinase.klifs_id"].to_list()
                kinases = remote.kinases.by_kinase_klifs_id(kinase_ids)
                protein.uniprot_id = kinases[
                    kinases["kinase.klifs_id"] == protein.klifs_kinase_id]["kinase.uniprot"].iloc[0]
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} requires a system with a protein having a 'klifs_kinase_id', " +
                "'uniprot_id' or 'pdb_id' attribute.")

        # identify DFG conformation of interest
        if not hasattr(protein, "dfg"):
            protein.dfg = None
        else:
            if protein.dfg not in ["in", "out", "out-like"]:
                raise NotImplementedError(
                    f"{self.__class__.__name__} requires a system with a protein having either no 'dfg' attribute" +
                    "or a 'dfg' attribute with a KLIFS specific DFG conformation ('in', 'out' or 'out-like').")

        # identify aC helix conformation of interest
        if not hasattr(protein, "ac_helix"):
            protein.ac_helix = None
        else:
            if protein.ac_helix not in ["in", "out", "out-like"]:
                raise NotImplementedError(
                    f"{self.__class__.__name__} requires a system with a protein having either no 'ac_helix' " +
                    "attribute or an 'ac_helix' attribute with a KLIFS specific alpha C helix conformation " +
                    "('in', 'out' or 'out-like').")

        # identify amino acid sequence of interest
        if not hasattr(protein, "sequence"):
            logging.debug(
                f"Retrieving kinase sequence details for UniProt entry {protein.uniprot_id} ...")
            protein.sequence = AminoAcidSequence.from_uniprot(protein.uniprot_id)

        return

    @staticmethod
    def _select_kinase_structure_by_pdb_id(
        pdb_id: str,
        klifs_kinase_id: int,
        chain_id: Union[str, None] = None,
        alternate_location: Union[str, None] = None,
    ):
        """
        Select a kinase structure via PDB identifier.
        Parameters
        ----------
        pdb_id: int
            The PDB identifier.
        klifs_kinase_id: int
            KLIFS kinase identifier.
        chain_id: str or None
            The chain of interest.
        alternate_location: str or None
            The alternate location of interest.
        Returns
        -------
        : pd.Series
            Details about the selected kinase structure.
        """
        from opencadd.databases.klifs import setup_remote

        logging.debug("Retrieve kinase structures from KLIFS matching the pdb of interest ...")
        remote = setup_remote()
        structures = remote.structures.by_structure_pdb_id(pdb_id, alternate_location, chain_id)
        structures = structures[structures["kinase.klifs_id"] == klifs_kinase_id]

        if len(structures) == 0:
            raise NotImplementedError(f"No structure found for PDB ID {pdb_id}, chain {chain_id} " +
                                      f"and alternate location {alternate_location}.")
        else:
            logging.debug("Picking structure with highest KLIFS quality score ...")
            structures = structures.sort_values(
                by=["structure.qualityscore", "structure.resolution", "structure.chain", "structure.alternate_model"],
                ascending=[False, True, True, True]
            )
            kinase_structure = structures.iloc[0]

        return kinase_structure

    @staticmethod
    def _select_kinase_structure_by_klifs_kinase_id(
        klifs_kinase_id: int,
        dfg: Union[str, None] = None,
        alpha_c_helix: Union[str, None] = None,
    ):
        """
        Select a kinase structure from KLIFS with the specified conformation.
        Parameters
        ----------
        klifs_kinase_id: int
            KLIFS kinase identifier.
        dfg: str
            The DFG conformation.
        alpha_c_helix: bool
            The alpha C helix conformation.
        Returns
        -------
        : pd.Series
            Details about the selected kinase structure.
        """
        from opencadd.databases.klifs import setup_remote

        logging.debug("Retrieve kinase structures from KLIFS matching the kinase of interest ...")
        remote = setup_remote()
        structures = remote.structures.by_kinase_klifs_id(klifs_kinase_id)

        logging.debug("Filtering KLIFS structures to match given kinase conformation ...")
        if dfg is not None:
            structures = structures[structures["structure.dfg"] == dfg]
        if alpha_c_helix is not None:
            structures = structures[structures["structure.ac_helix"] == alpha_c_helix]

        if len(structures) == 0:
            raise NotImplementedError(
                f"No structure available in DFG {dfg}/alpha C helix {alpha_c_helix} conformation."
            )
        else:
            logging.debug("Picking structure with highest KLIFS quality score ...")
            structures = structures.sort_values(
                by=["structure.qualityscore", "structure.resolution", "structure.chain", "structure.alternate_model"],
                ascending=[False, True, True, True]
            )
            kinase_structure = structures.iloc[0]

        return kinase_structure

    def _get_design_unit(
        self,
        protein_structure: oechem.OEGraphMol,
        structure_identifier: str,
        electron_density: Union[oegrid.OESkewGrid, None] = None,
        ligand_name: Union[str, None] = None,
        chain_id: Union[str, None] = None,
        alternate_location: Union[str, None] = None
    ) -> oechem.OEDesignUnit:
        """
        Get an OpenEye design unit from a protein ligand complex.
        Parameters
        ----------
        protein_structure: oechem.OEGraphMol
            An OpenEye molecule holding the protein that might be in complex with a ligand.
        structure_identifier: str
            A unique identifier describing the structure to prepare.
        electron_density: oegrid.OESkewGrid or None
            An OpenEye grid holding the electron density of the protein ligand complex.
        ligand_name: str or None
            Residue name of the ligand in complex with the protein structure.
        chain_id: str or None
            The chain of interest.
        alternate_location: str or None
            The alternate location of interest.
        Returns
        -------
        : oechem.OEDesignUnit
            The design unit.
        """
        from openeye import oechem

        from ..modeling.OEModeling import prepare_complex, prepare_protein
        from ..utils import LocalFileStorage

        if ligand_name == "-":
            ligand_name = None

        # generate unique design unit name
        design_unit_path = LocalFileStorage.featurizer_result(
            self.__class__.__name__,
            "_".join([
                structure_identifier,
                f"ligand{ligand_name}",
                f"chain{chain_id}",
                f"altloc{alternate_location}"
            ]),
            "oedu"
        )
        if not design_unit_path.is_file():
            logging.debug("Generating design unit ...")
            if ligand_name is None:
                design_unit = prepare_protein(
                    protein_structure,
                    chain_id=chain_id,
                    alternate_location=alternate_location,
                    cap_termini=False
                )
            else:
                design_unit = prepare_complex(
                    protein_structure,
                    electron_density=electron_density,
                    chain_id=chain_id,
                    alternate_location=alternate_location,
                    ligand_name=ligand_name,
                    cap_termini=False
                )
            logging.debug("Writing design unit ...")
            oechem.OEWriteDesignUnit(str(design_unit_path), design_unit)
        # re-reading design unit helps proper capping of e.g. 2itz
        # TODO: revisit, report bug
        logging.debug("Reading design unit from file ...")
        design_unit = oechem.OEDesignUnit()
        oechem.OEReadDesignUnit(str(design_unit_path), design_unit)

        return design_unit

    def _process_kinase_domain(
        self,
        kinase_structure: oechem.OEGraphMol,
        kinase_domain_sequence: Biosequence,
        chain_id: Union[str, None] = None
    ) -> oechem.OEGraphMol:
        """
        Process a kinase domain according to UniProt.
        Parameters
        ----------
        kinase_structure: oechem.OEGraphMol
            An OpenEye molecule holding the kinase structure to process.
        kinase_domain_sequence: Biosequence
            The kinase domain sequence with associated metadata.
        chain_id: str or None
            The chain of the kinase. Other chains will be deleted.
        Returns
        -------
        : oechem.OEGraphMol
            An OpenEye molecule holding the processed kinase structure.
        """

        from ..modeling.OEModeling import (
            select_chain,
            assign_caps,
            apply_deletions,
            apply_insertions,
            apply_mutations,
            delete_clashing_sidechains,
            delete_partial_residues,
            delete_short_protein_segments,
            renumber_structure
        )

        if chain_id:
            logging.debug(f"Deleting all chains but {chain_id} ...")
            kinase_structure = select_chain(kinase_structure, chain_id)

        logging.debug(f"Deleting residues with clashing side chains ...")  # e.g. 2j5f, 4wd5
        kinase_structure = delete_clashing_sidechains(kinase_structure)

        logging.debug("Deleting residues with missing atoms ...")
        kinase_structure = delete_partial_residues(kinase_structure)

        logging.debug("Deleting loose protein segments ...")
        kinase_structure = delete_short_protein_segments(kinase_structure)

        logging.debug("Applying deletions to kinase domain ...")
        kinase_structure = apply_deletions(kinase_structure, kinase_domain_sequence)

        logging.debug("Applying mutations to kinase domain ...")
        kinase_structure = apply_mutations(kinase_structure, kinase_domain_sequence)

        if self.loop_db:
            logging.debug("Applying insertions to kinase domain ...")
            kinase_structure = apply_insertions(kinase_structure, kinase_domain_sequence, self.loop_db)

        logging.debug("Renumbering residues ...")
        residue_numbers = self._get_kinase_residue_numbers(kinase_structure, kinase_domain_sequence)
        kinase_structure = renumber_structure(kinase_structure, residue_numbers)

        logging.debug("Checking kinase domain sequence termini ...")
        real_termini = []
        if kinase_domain_sequence.metadata["true_N_terminus"]:
            if kinase_domain_sequence.metadata["begin"] == residue_numbers[0]:
                real_termini.append(residue_numbers[0])
        if kinase_domain_sequence.metadata["true_C_terminus"]:
            if kinase_domain_sequence.metadata["end"] == residue_numbers[-1]:
                real_termini.append(residue_numbers[-1])
        if len(real_termini) == 0:
            real_termini = None

        logging.debug(f"Assigning caps except for real termini {real_termini} ...")
        processed_kinase_domain = assign_caps(kinase_structure, real_termini)

        return processed_kinase_domain

    @staticmethod
    def _get_kinase_residue_numbers(
        kinase_domain_structure: oechem.OEGraphMol,
        kinase_domain_sequence: Biosequence
    ) -> List[int]:
        """
        Get the canonical residue numbers of a kinase domain structure.
        Parameters
        ----------
        kinase_domain_structure: oechem.OEGraphMol
            The kinase domain structure.
        kinase_domain_sequence: Biosequence
            The canonical kinase domain sequence.
        Returns
        -------
        residue_number: list of int
            A list of canonical residue numbers in the same order as the residues in the given kinase domain structure.
        """
        from ..modeling.OEModeling import get_structure_sequence_alignment

        logging.debug("Aligning sequences ...")
        target_sequence, template_sequence = get_structure_sequence_alignment(
            kinase_domain_structure, kinase_domain_sequence)
        logging.debug(f"Template sequence:\n{template_sequence}")
        logging.debug(f"Target sequence:\n{target_sequence}")

        logging.debug("Generating residue numbers ...")
        residue_numbers = []
        residue_number = kinase_domain_sequence.metadata["begin"]
        for template_sequence_residue, target_sequence_residue in zip(
                template_sequence, target_sequence
        ):
            if template_sequence_residue != "-":
                if target_sequence_residue != "-":
                    residue_numbers.append(residue_number)
                residue_number += 1
            else:
                # TODO: This situation occurs if the given protein contains sequence segments that are not part of the
                #       canonical kinase domain sequence from UniProt. I don't think this will ever happen in the
                #       current implementation.
                raise NotImplementedError
        return residue_numbers


class OEKLIFSKinaseHybridDockingFeaturizer(OEKLIFSKinaseApoFeaturizer):
    """
    Given a System with exactly one kinase and one ligand,
    dock the ligand in the designated binding pocket.
    Parameters
    ----------
    loop_db: str or None
        The path to the loop database used by OESpruce to model missing loops.
    shape_overlay: bool
        If a shape overlay should be performed for selecting a ligand template
        in the hybrid docking protocol. Otherwise fingerprint similarity will
        be used.
    """
    import pandas as pd
    from openeye import oechem

    def __init__(
        self, loop_db: Union[str, None] = None, shape_overlay: bool = False
    ):
        super().__init__(loop_db)
        self.loop_db = loop_db
        self.shape_overlay = shape_overlay

    _SUPPORTED_TYPES = (ProteinLigandComplex,)

    @lru_cache(maxsize=100)
    def _featurize(self, system: ProteinLigandComplex) -> ProteinLigandComplex:
        """
        Perform hybrid docking in kinases using the OpenEye toolkit, the KLIFS database and thoughtful defaults.
        Parameters
        ----------
        system: ProteinLigandComplex
            A system object holding protein and ligand information.
        Returns
        -------
        protein_ligand_complex: ProteinLigandComplex
            The same system but with docked ligand.
        """
        from openeye import oechem

        from ..docking.OEDocking import create_hybrid_receptor, hybrid_docking
        from ..modeling.OEModeling import compare_molecules, read_smiles, get_expression_tags, delete_residue
        from ..utils import LocalFileStorage

        logging.debug("Interpreting kinase kinase of interest ...")
        self._interpret_kinase(system.protein)

        logging.debug("Interpreting ligand ...")
        if not hasattr(system.ligand, "smiles"):
            raise NotImplementedError(
                f"{self.__class__.__name__} requires a system with a ligand having a 'smiles' attribute."
            )

        logging.debug("Searching ligand template ...")  # TODO: naming problem with co-crystallized ligand in hybrid docking, see above
        ligand_template = self._select_ligand_template(
            system.protein.klifs_kinase_id,
            read_smiles(system.ligand.smiles),
            system.protein.dfg,
            system.protein.ac_helix
        )
        logging.debug(f"Selected {ligand_template['structure.pdb_id']} as ligand template ...")

        logging.debug("Searching kinase template ...")
        if ligand_template["kinase.klifs_id"] == system.protein.klifs_kinase_id:
            protein_template = ligand_template
        else:
            protein_template = self._select_kinase_structure_by_klifs_kinase_id(
                system.protein.klifs_kinase_id,
                ligand_template["structure.dfg"],
                ligand_template["structure.ac_helix"],
            )
        logging.debug(f"Selected {protein_template['structure.pdb_id']} as kinase template ...")

        logging.debug(f"Adding attributes to BaseProtein ...")  # TODO: bad idea in a library
        system.protein.pdb_id = protein_template["structure.pdb_id"]
        system.protein.path = LocalFileStorage.rcsb_structure_pdb(protein_template["structure.pdb_id"])
        system.protein.electron_density_path = LocalFileStorage.rcsb_electron_density_mtz(
            protein_template["structure.pdb_id"]
        )

        logging.debug(f"Interpreting system ...")
        ligand, kinase_structure, electron_density = self._interpret_system(system)

        logging.debug(f"Preparing kinase template structure of {protein_template['structure.pdb_id']} ...")
        design_unit = self._get_design_unit(
            kinase_structure,
            structure_identifier=protein_template["structure.pdb_id"],
            electron_density=electron_density,
            ligand_name=protein_template["ligand.expo_id"],
            chain_id=protein_template["structure.chain"],
            alternate_location=system.protein.alternate_location  # KLIFS alternate locations buggy, e.g. 3cs9
        )

        logging.debug(f"Preparing ligand template structure of {ligand_template['structure.pdb_id']} ...")
        prepared_ligand_template = self._prepare_ligand_template(ligand_template)

        logging.debug("Superposing kinase and ligand template ...")
        prepared_kinase, prepared_solvent = self._superpose_templates(
            design_unit, prepared_ligand_template, ligand_template
        )

        logging.debug("Deleting expression tags ...")
        expression_tags = get_expression_tags(kinase_structure)
        for expression_tag in expression_tags:
            prepared_kinase = delete_residue(
                prepared_kinase,
                chain_id=expression_tag["chain_id"],
                residue_name=expression_tag["residue_name"],
                residue_id=expression_tag["residue_id"]
            )

        logging.debug("Extracting ligand ...")
        split_options = oechem.OESplitMolComplexOptions()
        prepared_ligand_template = list(oechem.OEGetMolComplexComponents(
            prepared_ligand_template, split_options, split_options.GetLigandFilter())
        )[0]

        logging.debug("Processing kinase domain ...")
        processed_kinase_domain = self._process_kinase_domain(
            prepared_kinase,
            system.protein.sequence,
            protein_template["structure.chain"]
        )

        logging.debug("Checking for co-crystallized ligand ...")
        if (
            compare_molecules(ligand, prepared_ligand_template)
            and ligand_template["structure.pdb_id"] == protein_template["structure.pdb_id"]
        ):
            logging.debug(f"Found co-crystallized ligand ...")
            docking_pose = prepared_ligand_template
        else:
            logging.debug(f"Creating artificial hybrid receptor ...")
            hybrid_receptor = create_hybrid_receptor(
                processed_kinase_domain, prepared_ligand_template
            )
            logging.debug("Performing docking ...")
            docking_pose = hybrid_docking(hybrid_receptor, [ligand])[0]
            oechem.OEPerceiveResidues(docking_pose, oechem.OEPreserveResInfo_None)  # generate residue information

        logging.debug("Assembling components ...")
        kinase_ligand_complex = self._assemble_components(processed_kinase_domain, prepared_solvent, docking_pose)

        logging.debug("Updating pdb header ...")
        solvated_kinase = self._update_pdb_header(
            kinase_ligand_complex,
            protein_template["kinase.klifs_name"],
            system.ligand.name,
            [("COMPND", f"\tKinase template: {protein_template['structure.pdb_id']}"),
             ("COMPND", f"\tLigand template: {ligand_template['structure.pdb_id']}")]
        )

        logging.debug("Writing results ...")
        protein_path, ligand_path = self._write_results(
            solvated_kinase,
            "_".join([
                f"{protein_template['kinase.klifs_name']}",
                f"{protein_template['structure.pdb_id']}",
                f"chain{protein_template['structure.chain']}",
                f"altloc{system.protein.alternate_location}"  # KLIFS alternate locations buggy, e.g. 3cs9
            ]),
            system.ligand.name
        )

        logging.debug("Generating new system components")
        file_protein = FileProtein(path=protein_path)
        file_ligand = FileLigand(path=ligand_path)
        protein_ligand_complex = ProteinLigandComplex(
            components=[file_protein, file_ligand]
        )

        return protein_ligand_complex  # TODO: MDAnalysis objects

    def _select_ligand_template(
        self,
        klifs_kinase_id: int,
        ligand: oechem.OEGraphMol,
        dfg: Union[str or None],
        ac_helix: Union[str or None],
    ) -> pd.Series:
        """
        Select a kinase in complex with a ligand from KLIFS holding a ligand similar to the given SMILES, bound to
        a kinase similar to the kinase of interest and in the given KLIFS kinase conformation.
        Parameters
        ----------
        klifs_kinase_id: int
            KLIFS kinase identifier.
        ligand: oechem.OEGraphMol
            An OpenEye molecule holding the ligand that should be docked.
        dfg: str or None
            The KLIFS DFG conformation the ligand template should bind to.
        ac_helix: str or None
            The KLIFS alpha C helix conformation the ligand template should bind to.
        Returns
        -------
        : pd.Series
            Details about selected kinase and co-crystallized ligand.
        """
        from opencadd.databases.klifs import setup_remote

        logging.debug("Retrieve kinase information from KLIFS ...")
        remote = setup_remote()
        kinase_details = remote.kinases.by_kinase_klifs_id(klifs_kinase_id).iloc[0]

        logging.debug("Retrieve kinase structures from KLIFS for ligand template selection ...")
        structures = self._get_available_ligand_templates()

        if dfg:
            logging.debug(f"Filtering for ligands bound to a kinase in the DFG {dfg} conformation ...")
            structures = structures[structures["structure.dfg"] == dfg]

        if ac_helix:
            logging.debug(f"Filtering for ligands bound to a kinase in the alpha C helix {dfg} conformation ...")
            structures = structures[structures["structure.ac_helix"] == ac_helix]

        logging.debug("Storing SMILES in structures dataframe ...")
        structures = self._add_smiles_column(structures)

        logging.debug("Searching for identical co-crystallized ligands ...")
        identical_ligands = self._get_identical_ligand_indices(ligand, structures["smiles"])  # TODO: Takes surprisingly long

        if len(identical_ligands) > 0:
            logging.debug("Found identical co-crystallized ligands ...")
            structures = structures.iloc[identical_ligands]
            logging.debug("Searching for matching KLIFS kinase id ...")
            if structures["kinase.klifs_id"].isin([kinase_details["kinase.klifs_id"]]).any():
                logging.debug("Found matching KLIFS kinase id ...")
                structures = structures[
                    structures["kinase.klifs_id"].isin([kinase_details["kinase.klifs_id"]])
                ]
        else:
            if self.shape_overlay:
                logging.debug("Filtering for most similar ligands according to their shape overlay ...")
                structures = self._filter_for_similar_ligands_3d(ligand, structures)
            else:
                logging.debug("Filtering for most similar ligands according to their fingerprints ...")
                structures = self._filter_for_similar_ligands_2d(ligand, structures)

        logging.debug("Filtering for most similar kinase pockets ...")
        structures = self._filter_for_similar_kinase_pockets(kinase_details["kinase.pocket"], structures)

        logging.debug("Picking structure with highest KLIFS quality ...")
        structure_for_ligand = structures.iloc[0]

        return structure_for_ligand

    @staticmethod
    def _get_available_ligand_templates():
        """
        Get available ligand templates from KLIFS.
        Returns
        -------
        : pd.DataFrame
            A pandas dataframe containing information about available ligand templates.
        """
        from opencadd.databases.klifs import setup_remote

        logging.debug("Retrieving available KLIFS entries ...")
        remote = setup_remote()
        structures = remote.structures.all_structures()

        logging.debug("Filtering KLIFS entries ...")
        structures = structures[structures["ligand.expo_id"] != "-"]  # orthosteric ligand
        structures = structures.groupby("structure.pdb_id").filter(
            lambda x: len(set(x["ligand.expo_id"])) == 1
        )  # single orthosteric ligand
        structures = structures[
            structures["ligand_allosteric.expo_id"] == "-"
            ]  # no allosteric ligand
        structures = structures[structures["structure.dfg"] != 'na']  # no missing kinase conformations
        structures = structures[structures["structure.ac_helix"] != 'na']

        logging.debug("Sorting KLIFS entries by quality ...")
        # keep entry with highest quality score (alt 'A' preferred over alt 'B', chain 'A' preferred over 'B')
        structures = structures.sort_values(
            by=["structure.qualityscore", "structure.resolution", "structure.chain", "structure.alternate_model"],
            ascending=[False, True, True, True]
        )

        logging.debug("Filtering for highest quality KLIFS entry per PDB code ...")
        structures = structures.groupby("structure.pdb_id").head(1)

        return structures

    @staticmethod
    def _add_smiles_column(structures: pd.DataFrame) -> pd.DataFrame:
        """
        Add SMILES column to a DataFrame containing KLIFS entries.
        Parameters
        ----------
        structures: pd.DataFrame
            A DataFrame containing KLIFS entries.
        Returns
        -------
        : pd.DataFrame
            The input DataFrame with an additional SMILES column.
        """
        import json

        from ..modeling.OEModeling import smiles_from_pdb
        from ..utils import LocalFileStorage

        logging.debug("Retrieving smiles information from PDB ...")
        if LocalFileStorage.pdb_smiles_json().is_file():
            logging.debug("Reading local PDB SMILES dictionary ...")
            with open(LocalFileStorage.pdb_smiles_json(), "r") as rf:
                pdb_to_smiles = json.load(rf)
        else:
            logging.debug("Initiating new PDB to SMILES dictionary ...")
            pdb_to_smiles = {}

        logging.debug("Retrieving SMILES for unknown ligands ...")
        pdb_to_smiles.update(
            smiles_from_pdb(set(structures["ligand.expo_id"]) - set(pdb_to_smiles.keys()))
        )

        logging.debug("Saving local PDB SMILES dictionary ...")
        with open(LocalFileStorage.pdb_smiles_json(), "w") as wf:
            json.dump(pdb_to_smiles, wf)

        logging.debug("Adding SMILES to DataFrame ...")
        smiles_column = []
        for ligand_id in structures["ligand.expo_id"]:
            if ligand_id in pdb_to_smiles.keys():
                smiles_column.append(pdb_to_smiles[ligand_id])
            else:
                smiles_column.append(None)
        structures["smiles"] = smiles_column

        logging.debug("Removing structures without a SMILES ...")
        structures = structures[structures["smiles"].notnull()]

        return structures

    @staticmethod
    def _get_identical_ligand_indices(ligand: oechem.OEGraphMol, smiles_iterable: Iterable[str]) -> List[int]:
        """
        Get the indices of the SMILES matching the given ligand.
        Parameters
        ----------
        ligand: oechem.OEGraphMol
            An OpenEye molecule holding the ligand to dock.
        smiles_iterable: iterable of str
            An iterable of SMILES strings representing the molecules to compare with ligand.
        Returns
        -------
        : list of int
            The indices of matching SMILES strings.
        """
        from ..modeling.OEModeling import read_smiles, compare_molecules

        identical_ligand_indices = []
        for i, complex_ligand in enumerate(smiles_iterable):
            if compare_molecules(ligand, read_smiles(complex_ligand)):
                identical_ligand_indices.append(i)

        return identical_ligand_indices

    @staticmethod
    def _filter_for_similar_ligands_3d(ligand: oechem.OEGraphMol, structures: pd.DataFrame) -> pd.DataFrame:
        """
        Filter KLIFS structures for similar ligands according to a shape overlay.
        Parameters
        ----------
        ligand: oechem.OEGraphMol
            An OpenEye molecule holding the ligand to dock.
        structures: pd.DataFrame
            A DataFrame containing KLIFS entries.
        Returns
        -------
        : pd.DataFrame
            The input DataFrame filtered for KLIFS entries with most similar ligands.
        """
        from ..modeling.OEModeling import get_klifs_ligand, generate_reasonable_conformations, overlay_molecules

        logging.debug("Retrieving resolved structures of orthosteric ligands ...")
        complex_ligands = [get_klifs_ligand(structure_id) for structure_id in structures["structure.klifs_id"]]

        logging.debug("Generating reasonable conformations of ligand of interest ...")
        conformations_ensemble = generate_reasonable_conformations(ligand)

        logging.debug("Overlaying molecules ...")
        overlay_scores = []
        for conformations in conformations_ensemble:
            overlay_scores += [
                [i, overlay_molecules(complex_ligand, conformations, False)]
                for i, complex_ligand in enumerate(complex_ligands)
            ]

        # if maximal score is 1.73, threshold is set to 1.53
        overlay_score_threshold = max([score[1] for score in overlay_scores]) - 0.2

        logging.debug("Picking structures with most similar ligands ...")
        structures = structures.iloc[[score[0] for score in overlay_scores if score[1] >= overlay_score_threshold]]

        return structures

    @staticmethod
    def _filter_for_similar_ligands_2d(ligand: oechem.OEGraphMol, structures: pd.DataFrame) -> pd.DataFrame:
        """
        Filter KLIFS structures for similar ligands according to a fingerprint comparison.
        Parameters
        ----------
        ligand: oechem.OEGraphMol
            An OpenEye molecule holding the ligand to dock.
        structures: pd.DataFrame
            A DataFrame containing KLIFS entries.
        Returns
        -------
        : pd.DataFrame
            The input DataFrame filtered for KLIFS entries with most similar ligands.
        """
        import pandas as pd
        from openeye import oechem
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem, DataStructs

        RDLogger.DisableLog("rdApp.*")  # disable RDKit logging

        logging.debug("Converting OpenEye molecule to RDKit molecule ...")
        ligand = Chem.MolFromSmiles(oechem.OEMolToSmiles(ligand))

        logging.debug("Converting SMILES to RDKit molecules ...")
        rdkit_molecules = [Chem.MolFromSmiles(smiles) for smiles in structures["smiles"]]

        logging.debug("Adding RDKit molecules to dataframe...")
        structures["rdkit_molecules"] = rdkit_molecules

        logging.debug("Removing KLIFS entries without valid RDKit molecule ...")
        structures = structures[structures["rdkit_molecules"].notnull()]

        logging.debug("Adding Feature Morgan fingerprint to dataframe...")
        pd.options.mode.chained_assignment = None  # otherwise next line would raise a warning
        structures["rdkit_fingerprint"] = [
            AllChem.GetMorganFingerprint(rdkit_molecule, 2, useFeatures=True)
            for rdkit_molecule in structures["rdkit_molecules"]
        ]

        logging.debug("Generating Feature Morgan fingerprint of ligand ...")
        ligand_fingerprint = AllChem.GetMorganFingerprint(ligand, 2, useFeatures=True)

        logging.debug("Calculating dice similarity between fingerprints ...")
        fingerprint_similarities = [
            [i, DataStructs.DiceSimilarity(ligand_fingerprint, fingerprint)]
            for i, fingerprint in enumerate(structures["rdkit_fingerprint"])
        ]

        # if maximal score is 0.87, threshold is set to 0.77
        fingerprint_similarity_threshold = max([similarity[1] for similarity in fingerprint_similarities]) - 0.1

        logging.debug("Picking structures with most similar ligands ...")
        structures = structures.iloc[
            [similarity[0]
             for similarity in fingerprint_similarities
             if similarity[1] >= fingerprint_similarity_threshold]
        ]

        return structures

    @staticmethod
    def _filter_for_similar_kinase_pockets(reference_pocket, structures):
        """
        Filter KLIFS structures for most similar kinase pockets compared to the reference pocket.
        Parameters
        ----------
        reference_pocket: str
            The kinase pocket sequence the structures should be compared to.
        structures: pd.DataFrame
            A DataFrame containing KLIFS entries.
        Returns
        -------
        : pd.DataFrame
            The input DataFrame filtered for KLIFS entries with most similar kinase pockets.
        """
        from ..modeling.OEModeling import sequence_similarity

        logging.debug("Calculating string similarity between KLIFS pockets ...")
        pocket_similarities = [sequence_similarity(structure_pocket, reference_pocket) for structure_pocket in
                               structures["structure.pocket"]]

        logging.debug("Adding pocket similarity to dataframe...")
        structures["pocket_similarity"] = pocket_similarities

        # if maximal possible score is 498, similarity threshold is corrected by 49.8
        threshold_correction = sequence_similarity(reference_pocket, reference_pocket) / 10
        pocket_similarity_threshold = max(pocket_similarities) - threshold_correction

        logging.debug("Picking structures with most similar kinase pockets ...")
        structures = structures[structures["pocket_similarity"] >= pocket_similarity_threshold]

        return structures

    @staticmethod
    def _prepare_ligand_template(ligand_template: pd.Series) -> oechem.OEGraphMol:
        """
        Prepare a PDB structure containing the ligand template of interest.
        Parameters
        ----------
        ligand_template: pd.Series
            A data series containing entries 'structure.pdb_id', 'structure.chain', 'ligand.expo_id' and
            'structure.alternate_model'.
        Returns
        -------
        : oechem.OEGraphMol
            An OpenEye molecule holding the prepared ligand template structure.
        """
        from openeye import oechem

        from ..modeling.OEModeling import read_molecules, select_chain, select_altloc, remove_non_protein
        from ..utils import FileDownloader

        logging.debug("Interpreting structure ...")
        ligand_template_structure = PDBProtein(ligand_template["structure.pdb_id"])
        if not ligand_template_structure.path.is_file():
            logging.debug(
                f"Downloading PDB entry {ligand_template['structure.pdb_id']} ..."
            )
            FileDownloader.rcsb_structure_pdb(ligand_template["structure.pdb_id"])
        logging.debug("Reading structure ...")
        ligand_template_structure = read_molecules(ligand_template_structure.path)[0]

        logging.debug("Selecting chain ...")
        ligand_template_structure = select_chain(ligand_template_structure, ligand_template["structure.chain"])

        logging.debug("Selecting alternate location ...")
        ligand_template_structure = select_altloc(
            ligand_template_structure, ligand_template["structure.alternate_model"]
        )

        logging.debug("Removing everything but protein, water and ligand of interest ...")
        ligand_template_structure = remove_non_protein(
            ligand_template_structure, exceptions=[ligand_template["ligand.expo_id"]], remove_water=False
        )

        logging.debug("Adding hydrogens ...")
        oechem.OEPlaceHydrogens(ligand_template_structure)

        return ligand_template_structure

    @staticmethod
    def _superpose_templates(
        design_unit: oechem.OEDesignUnit,
        ligand_template_structure: oechem.OEGraphMol,
        ligand_template: pd.Series
    ) -> Tuple[oechem.OEGraphMol, oechem.OEGraphMol]:
        """
        Superpose the kinase domain from the design unit to the given ligand template structure. The superposed kinase
        domain will be returned with kinase domain and solvent separated.
        Parameters
        ----------
        design_unit: oechem.OEDesignUnit
            The OpenEye design unit containing the kinase domain.
        ligand_template_structure: oechem.OEGraphMol
            An OpenEye molecule holding the ligand template structure.
        ligand_template: pd.Series
            A data series containing entries 'structure.chain' and 'structure.klifs_id'.
        Returns
        -------
        kinase_domain: oechem.OEGraphMol
            The superposed kinase domain without solvent.
        solvent: oechem.OEGraphMol
            The solvent of the superposed kinase domain.
        """

        from opencadd.databases.klifs import setup_remote
        from openeye import oechem

        from ..modeling.OEModeling import superpose_proteins

        logging.debug("Extracting protein and solvent ...")
        solvated_kinase_domain = oechem.OEGraphMol()
        design_unit.GetComponents(
            solvated_kinase_domain, oechem.OEDesignUnitComponents_Protein | oechem.OEDesignUnitComponents_Solvent
        )

        logging.debug("Retrieving KLIFS kinase pocket residues ...")
        remote = setup_remote()
        pocket = remote.coordinates.to_dataframe(ligand_template["structure.klifs_id"], entity="pocket")
        pocket_residues = set(pocket["residue.name"] + pocket["residue.id"])

        logging.debug("Superposing structure on kinase domain ...")
        solvated_kinase_domain = superpose_proteins(
            ligand_template_structure, solvated_kinase_domain, pocket_residues, ligand_template["structure.chain"]
        )

        logging.debug("Separating solvent from kinase domain ...")
        kinase_domain, solvent = oechem.OEGraphMol(), oechem.OEGraphMol()
        oechem.OESplitMolComplex(
            oechem.OEGraphMol(), kinase_domain, solvent, oechem.OEGraphMol(), solvated_kinase_domain
        )

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
        )
        oechem.OEPerceiveResidues(kinase_domain, preserved_info)
        oechem.OEPerceiveResidues(solvent, preserved_info)

        return kinase_domain, solvent
