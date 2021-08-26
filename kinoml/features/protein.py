"""
Featurizers that mostly concern protein-based models
"""
from __future__ import annotations
import numpy as np
from collections import Counter
from functools import lru_cache
import logging
from pathlib import Path
from typing import Union, Tuple, Iterable, List

from .core import ParallelBaseFeaturizer, BaseOneHotEncodingFeaturizer
from ..core.systems import System, ProteinSystem, ProteinLigandComplex
from ..core.proteins import AminoAcidSequence


class AminoAcidCompositionFeaturizer(ParallelBaseFeaturizer):

    """
    Featurizes the protein using the composition of the residues
    in the binding site.
    """

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    ALPHABET = AminoAcidSequence.ALPHABET

    def _retrieve_sequence(self, system: System) -> str:
        for comp in system.components:
            if isinstance(comp, AminoAcidSequence):
                return comp.sequence

class OESpruceProteinStructureFeaturizer(ParallelBaseFeaturizer):


    """
    Given a System with exactly one protein and one ligand, prepare the binding pocket without a ligand.
    
    We assume that a smiles and file-based System object will be passed: this means we will have a System.components with FileProtein and FileLigand or SmilesLigand. The file itself could be a URL.
    
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
        Assign the predominant ionization state of the molecules to dock at pH ~7.4.
        If False, the ionization state of the input molecules will be conserved.
    """

    from openeye import oechem, oegrid

    def __init__(
            self,
            loop_db: Union[str, None] = None,
            cache_dir: Union[str, Path, None] = None,
            output_dir: Union[str, Path, None] = None,
            pKa_norm: bool = True,
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
        self.pKa_norm = pKa_norm

    _SUPPORTED_TYPES = (ProteinLigandComplex,)

    @lru_cache(maxsize=100)
    def _featurize_one(self, system: ProteinLigandComplex) -> universe:
        """
        Prepare for hybrid docking with the OpenEye toolkit and thoughtful defaults but without doing the docking.
        
        Parameters
        ----------
        systems: iterable of ProteinLigandComplex
            A list of System objects holding protein and ligand information.
        Returns
        -------
        : universe
            An MDAnalysis universe of the featurized system.
        """
        from openeye import oechem

        from ..docking.OEDocking import create_hybrid_receptor, hybrid_docking

        logging.debug("Interpreting system ...")
        ligand, protein, electron_density = self._interpret_system(system)

        logging.debug("Preparing protein ligand complex ...")
        design_unit = self._get_design_unit(protein, system.protein.name, electron_density)

        logging.debug("Extracting components ...")
        # TODO: rename prepared_ligand
        prepared_protein, prepared_solvent, prepared_ligand = self._get_components(design_unit)

#        logging.debug("Creating hybrid receptor ...")
#        # TODO: takes quite long, should save this somehow
#        hybrid_receptor = create_hybrid_receptor(prepared_protein, prepared_ligand)

#        logging.debug("Performing docking ...")
#        docking_pose = hybrid_docking(hybrid_receptor, [ligand], pKa_norm=self.pKa_norm)[0]
#        # generate residue information
#        oechem.OEPerceiveResidues(docking_pose, oechem.OEPreserveResInfo_None)

        logging.debug("Assembling components ...")
        protein_ligand_complex = self._assemble_components(prepared_protein, prepared_solvent)

        logging.debug("Updating pdb header ...")
        protein_ligand_complex = self._update_pdb_header(
            protein_ligand_complex,
            protein_name=system.protein.name,
            ligand_name=system.ligand.name,
        )

        logging.debug("Writing results ...")
        file_path = self._write_results(
            protein_ligand_complex,
            system.protein.name,
            system.ligand.name,
        )

        logging.debug("Generating new MDAnalysis universe ...")
        structure = ProteinStructure.from_file(file_path)

        if not self.output_dir:
            logging.debug("Removing structure file ...")
            file_path.unlink()

        return structure

    def _interpret_system(
            self,
            system: Union[ProteinSystem, ProteinLigandComplex],
    ) -> Tuple[
        Union[oechem.OEGraphMol, None],
        Union[oechem.OEGraphMol, None],
        Union[oegrid.OESkewGrid, None]
    ]:
        """
        Interpret the given system components and retrieve OpenEye objects holding protein and electron density.
        
        Parameters
        ----------
        system: ProteinLigandComplex
            The system to featurize.
        
        Returns
        -------
        : tuple of oechem.OEGraphMol, oechem.OEGraphMol and oegrid.OESkewGrid or None
            OpenEye objects holding ligand, protein and electron density
        """
        from ..modeling.OEModeling import (
            read_smiles,
            read_molecules,
            read_electron_density,
        )
        from ..utils import FileDownloader, LocalFileStorage

        ligand = None
        if hasattr(system, "ligand"):
            logging.debug("Interpreting ligand ...")
            if hasattr(system.ligand, "path"):
                logging.debug(f"Loading ligand from {system.ligand.path} ...")
                ligand = read_molecules(system.ligand.path)[0]

            else:
                logging.debug("Loading ligand from SMILES string ...")
                ligand = read_smiles(system.ligand.to_smiles())

        logging.debug("Interpreting protein ...")
        if hasattr(system.protein, "pdb_id"):
            system.protein.path = LocalFileStorage.rcsb_structure_pdb(
                system.protein.pdb_id, self.cache_dir
            )
            if not system.protein.path.is_file():
                logging.debug(
                    f"Downloading protein structure {system.protein.pdb_id} from PDB ..."
                )
                FileDownloader.rcsb_structure_pdb(system.protein.pdb_id, self.cache_dir)
        logging.debug(f"Reading protein structure from {system.protein.path} ...")
        protein = read_molecules(system.protein.path)[0]

        logging.debug("Interpreting electron density ...")
        electron_density = None
        # if hasattr(system.protein, "pdb_id"):
        #     system.protein.electron_density_path = LocalFileStorage.rcsb_electron_density_mtz(
        #         system.protein.pdb_id, cache_dir
        #     )
        #     if not system.protein.electron_density_path.is_file():
        #         logging.debug(f"Downloading electron density for structure {system.protein.pdb_id} from PDB ...")
        #         FileDownloader.rcsb_electron_density_mtz(system.protein.pdb_id, self.cache_dir)
        #
        # if hasattr(system.protein, "electron_density_path"):
        # TODO: Kills Kernel for some reason
        #    electron_density = read_electron_density(system.protein.electron_density_path)

        return ligand, protein, electron_density

    def _get_design_unit(
            self,
            complex_structure: oechem.OEGraphMol,
            design_unit_identifier: str,
            electron_density: Union[oegrid.OESkewGrid, None],
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
            f"{design_unit_identifier}_design_unit", "oedu",
            self.cache_dir,
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

class OEKLIFSKinaseProteinStructureFeaturizer(OESpruceProteinStructureFeaturizer):

    """
    Given a System with exactly one kinase prepare an apo kinase.
    """

    import pandas as pd
    from openeye import oechem, oegrid

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    _SUPPORTED_TYPES = (ProteinSystem,)

    def _pre_featurize(self, systems: Iterable[ProteinSystem]) -> None:
        """
        Retrieve relevant data from KLIFS and store locally.
        """
        self._create_klifs_structure_db()
        self._create_klifs_kinase_db()
        return

    def _create_klifs_structure_db(self, retrieve_pocket_resids=False):
        """
        Retrieve structure data from KLIFS and store locally.
        
        Parameters
        ----------
        retrieve_pocket_resids: bool
            If pocket residue IDs should be retrieved (needed for docking).
        """
        from opencadd.databases.klifs import setup_remote
        import numpy as np
        import pandas as pd

        from ..utils import LocalFileStorage

        remote = setup_remote()
        logging.debug("Retrieving all structures from KLIFS ...")
        available_structures = remote.structures.all_structures()
        available_structures["structure.pocket_resids"] = np.NaN

        klifs_structure_db_path = LocalFileStorage.klifs_structure_db(self.cache_dir)
        # checking for existing database, since retrieving pocket residue IDs takes long
        if klifs_structure_db_path.is_file():
            logging.debug("Loading local KLIFS structure database ...")
            klifs_structure_db = pd.read_csv(klifs_structure_db_path)
        else:
            logging.debug("Initializing local KLIFS structure database ...")
            klifs_structure_db = available_structures.copy()

        logging.debug("Searching new structures ...")
        new_structures = available_structures[
            ~available_structures["structure.klifs_id"].isin(
                klifs_structure_db["structure.klifs_id"]
            )
        ]

        if len(new_structures) > 0:
            logging.debug("Adding new structures to database ...")
            klifs_structure_db = klifs_structure_db.append(
                available_structures[available_structures["structure.klifs_id"].isin(
                    new_structures
                )]
            )

        if retrieve_pocket_resids:
            logging.debug("Adding KLIFS pocket residue IDs ...")
            structures_wo_pocket_resids = klifs_structure_db[
                klifs_structure_db["structure.pocket_resids"].isna()
            ]
            for structure_klifs_id in structures_wo_pocket_resids["structure.klifs_id"]:
                pocket = remote.pockets.by_structure_klifs_id(structure_klifs_id)
                if any(pd.isnull(pocket["residue.id"])):
                    pocket_ids = ""
                else:
                    pocket_ids = " ".join(  # filter out missing residues defined as "_"
                        [residue_id for residue_id in pocket["residue.id"] if residue_id != "_"]
                    )
                klifs_structure_db.loc[
                    (klifs_structure_db["structure.klifs_id"] == structure_klifs_id),
                    "structure.pocket_resids"
                ] = pocket_ids
            logging.debug("Removing entries with missing pocket residue IDs ...")
            klifs_structure_db = klifs_structure_db[
                klifs_structure_db["structure.pocket_resids"] != ""
                ]

        logging.debug("Saving KLIFS data locally ...")
        klifs_structure_db.to_csv(klifs_structure_db_path, index=False)

        return

    def _create_klifs_kinase_db(self):
        """
        Retrieve kinase data from KLIFS and store locally.
        """
        from opencadd.databases.klifs import setup_remote

        from ..utils import LocalFileStorage

        remote = setup_remote()
        logging.debug("Retrieving all kinases from KLIFS ...")
        klifs_kinase_ids = remote.kinases.all_kinases()["kinase.klifs_id"].to_list()
        klifs_kinase_db = remote.kinases.by_kinase_klifs_id(klifs_kinase_ids)

        logging.debug("Saving KLIFS data locally ...")
        klifs_kinase_db.to_csv(LocalFileStorage.klifs_kinase_db(self.cache_dir), index=False)

        return

    @lru_cache(maxsize=100)
    def _featurize_one(self, system: ProteinSystem) -> universe:
        """
        Prepare a kinase using the OpenEye toolkit, the KLIFS database and thoughtful defaults.
        
        Parameters
        ----------
        system: ProteinSystem
            A system object holding protein information.
        
        Returns
        -------
        : universe
            An MDAnalysis universe of the featurized system.
        """
        import MDAnalysis as mda

        from ..modeling.OEModeling import (
            get_expression_tags,
            delete_residue,
            select_chain,
        )
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
            system.protein.path = LocalFileStorage.rcsb_structure_pdb(
                kinase_details["structure.pdb_id"], self.cache_dir
            )
            system.protein.electron_density_path = LocalFileStorage.rcsb_electron_density_mtz(
                kinase_details["structure.pdb_id"], self.cache_dir
            )

        logging.debug("Interpreting system ...")
        kinase_structure, electron_density = self._interpret_system(system)[1:]

        logging.debug(f"Preparing kinase template structure of {kinase_details['structure.pdb_id']} ...")
        try:
            design_unit = self._get_design_unit(
                kinase_structure,
                structure_identifier=kinase_details["structure.pdb_id"],
                electron_density=electron_density,
                ligand_name=kinase_details["ligand.expo_id"],
                chain_id=kinase_details["structure.chain"],
                alternate_location=kinase_details["structure.alternate_model"],
            )
        except ValueError:
            logging.debug(
                f"Could not generate design unit for PDB entry "
                f"{kinase_details['structure.pdb_id']} with alternate location "
                f"{kinase_details['structure.alternate_model']} and chain ID " 
                f"{kinase_details['structure.chain']}. Returning empty universe ..."
            )
            return mda.Universe.empty(0)

        logging.debug("Extracting kinase and solvent from design unit ...")
        prepared_kinase, prepared_solvent = self._get_components(design_unit)[:-1]

        logging.debug("Selecting chain of solvent ...")
        try:
            prepared_solvent = select_chain(prepared_solvent, kinase_details["structure.chain"])
        except ValueError:
            logging.debug(f"No solvent in chain {kinase_details['structure.chain']}...")
            pass

        logging.debug("Deleting expression tags ...")
        expression_tags = get_expression_tags(kinase_structure)
        for expression_tag in expression_tags:
            try:
                prepared_kinase = delete_residue(
                    prepared_kinase,
                    chain_id=expression_tag["chain_id"],
                    residue_name=expression_tag["residue_name"],
                    residue_id=expression_tag["residue_id"]
                )
            except ValueError:
                pass  # wrong chain or not resolved

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
        file_path = self._write_results(
            solvated_kinase,
            "_".join([
                f"{kinase_details['kinase.klifs_name']}",
                f"{kinase_details['structure.pdb_id']}",
                f"chain{kinase_details['structure.chain']}",
                f"altloc{kinase_details['structure.alternate_model']}"
            ]),
            None
        )

        logging.debug("Generating new MDAnalysis universe ...")
        structure = ProteinStructure.from_file(file_path)

        if not self.output_dir:
            logging.debug("Removing structure file ...")
            file_path.unlink()

        return structure

    def _interpret_kinase(self, protein: ProteinSystem):
        """
        Interpret the kinase information stored in the given Protein object.
        
        Parameters
        ----------
        protein: Protein
            The Protein object.
        """
        import pandas as pd

        from ..core.sequences import AminoAcidSequence
        from ..utils import LocalFileStorage

        klifs_structures = pd.read_csv(LocalFileStorage.klifs_structure_db(self.cache_dir))
        klifs_kinases = pd.read_csv(LocalFileStorage.klifs_kinase_db(self.cache_dir))

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
                structures = klifs_structures[
                    klifs_structures["structure.pdb_id"] == protein.pdb_id
                    ]
                if protein.alternate_location:
                    structures = structures[
                        structures["structure.alternate_model"] == protein.alternate_location
                        ]
                if protein.chain_id:
                    structures = structures[
                        structures["structure.chain"] == protein.chain_id
                        ]
                protein.klifs_kinase_id = structures["kinase.klifs_id"].iloc[0]
            # if KLIFS kinase ID is not given, query by UniProt ID
            if not hasattr(protein, "klifs_kinase_id"):
                logging.debug("Converting UniProt ID to KLIFS kinase ID ...")
                protein.klifs_kinase_id = klifs_kinases[
                    klifs_kinases["kinase.uniprot"] == protein.uniprot_id
                ]["kinase.klifs_id"].iloc[0]
            # if UniProt ID is not given, query by KLIFS kinase ID
            if not hasattr(protein, "uniprot_id"):
                logging.debug("Converting KLIFS kinase ID to UniProt ID  ...")
                protein.uniprot_id = klifs_kinases[
                    klifs_kinases["kinase.klifs_id"] == protein.klifs_kinase_id
                    ]["kinase.uniprot"].iloc[0]
        else:
            text = (
                f"{self.__class__.__name__} requires a system with a protein having a "
                "'klifs_kinase_id', 'uniprot_id' or 'pdb_id' attribute.")
            logging.debug("Exception: " + text)
            raise NotImplementedError(text)

        # identify DFG conformation of interest
        if not hasattr(protein, "dfg"):
            protein.dfg = None
        else:
            if protein.dfg not in ["in", "out", "out-like"]:
                text = (
                    f"{self.__class__.__name__} requires a system with a protein having either no "
                    "'dfg' attribute or a 'dfg' attribute with a KLIFS specific DFG conformation "
                    "('in', 'out' or 'out-like')."
                )
                logging.debug("Exception: " + text)
                raise NotImplementedError(text)

        # identify aC helix conformation of interest
        if not hasattr(protein, "ac_helix"):
            protein.ac_helix = None
        else:
            if protein.ac_helix not in ["in", "out", "out-like"]:
                text = (
                    f"{self.__class__.__name__} requires a system with a protein having either no "
                    "'ac_helix' attribute or an 'ac_helix' attribute with a KLIFS specific alpha C"
                    " helix conformation ('in', 'out' or 'out-like')."
                )
                logging.debug("Exception: " + text)
                raise NotImplementedError(text)

        # identify amino acid sequence of interest
        if not hasattr(protein, "sequence"):
            logging.debug(
                f"Retrieving kinase sequence details for UniProt entry {protein.uniprot_id} ...")
            protein.sequence = AminoAcidSequence.from_uniprot(protein.uniprot_id)

        return  # TODO: What to do if kinase not in KLIFS?

    def _select_kinase_structure_by_pdb_id(
        self,
        pdb_id: str,
        klifs_kinase_id: int,
        chain_id: Union[str, None] = None,
        alternate_location: Union[str, None] = None,
    ) -> pd.Series:
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
        import pandas as pd

        from ..utils import LocalFileStorage

        logging.debug("Searching kinase structures from KLIFS matching the pdb of interest ...")
        klifs_structures = pd.read_csv(LocalFileStorage.klifs_structure_db(self.cache_dir))
        structures = klifs_structures[
            klifs_structures["structure.pdb_id"] == pdb_id
            ]
        structures = structures[
            structures["kinase.klifs_id"] == klifs_kinase_id
            ]
        if alternate_location:
            structures = structures[
                structures["structure.alternate_model"] == alternate_location
                ]
        if chain_id:
            structures = structures[
                structures["structure.chain"] == chain_id
                ]

        if len(structures) == 0:
            text = (
                f"No structure found for PDB ID {pdb_id}, chain {chain_id} and alternate location "
                f"{alternate_location}."
            )
            logging.debug("Exception: " + text)
            raise NotImplementedError(text)
        else:
            logging.debug("Picking structure with highest KLIFS quality score ...")
            structures = structures.sort_values(
                by=[
                    "structure.qualityscore",
                    "structure.resolution",
                    "structure.chain",
                    "structure.alternate_model"
                ],
                ascending=[False, True, True, True]
            )
            kinase_structure = structures.iloc[0]

        return kinase_structure

    def _select_kinase_structure_by_klifs_kinase_id(
            self,
            klifs_kinase_id: int,
            dfg: Union[str, None] = None,
            alpha_c_helix: Union[str, None] = None,
    ) -> pd.Series:
        """
        Select a kinase structure from KLIFS with the specified conformation.
        
        Parameters
        ----------
        klifs_kinase_id: int
            KLIFS kinase identifier.
        dfg: str or None, default=None
            The DFG conformation.
        alpha_c_helix: str or None, default=None
            The alpha C helix conformation.
        
        Returns
        -------
        : pd.Series
            Details about the selected kinase structure.
        """
        import pandas as pd

        from ..utils import LocalFileStorage

        logging.debug("Getting kinase reference pocket ...")
        klifs_kinases = pd.read_csv(LocalFileStorage.klifs_kinase_db(self.cache_dir))
        reference_pocket = klifs_kinases[
            klifs_kinases["kinase.klifs_id"] == klifs_kinase_id
            ]["kinase.pocket"].iloc[0]
        reference_pocket = reference_pocket.replace("_", "")

        logging.debug("Searching kinase structures from KLIFS matching the kinase of interest ...")
        klifs_structures = pd.read_csv(LocalFileStorage.klifs_structure_db(self.cache_dir))
        structures = klifs_structures[
            klifs_structures["kinase.klifs_id"] == klifs_kinase_id
            ]

        logging.debug("Filtering KLIFS structures to match given kinase conformation ...")
        if dfg is not None:
            structures = structures[structures["structure.dfg"] == dfg]
        if alpha_c_helix is not None:
            structures = structures[structures["structure.ac_helix"] == alpha_c_helix]

        if len(structures) == 0:
            # TODO: Use homology modeling or something similar
            text = (
                f"No structure available in DFG {dfg}/alpha C helix {alpha_c_helix} conformation."
            )
            logging.debug("Exception: " + text)
            raise NotImplementedError(text)
        else:
            structures = self._add_kinase_pocket_similarity(reference_pocket, structures)
            logging.debug("Sorting kinase structures by quality ...")
            structures = structures.sort_values(
                by=[
                    "pocket_similarity",  # detects missing residues and mutations
                    "structure.qualityscore",
                    "structure.resolution",
                    "structure.chain",
                    "structure.alternate_model"
                ],
                ascending=[False, False, True, True, True]
            )

            kinase_structure = structures.iloc[0]

        return kinase_structure

    @staticmethod
    def _add_kinase_pocket_similarity(
            reference_pocket: str, structures: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add a column to the input DataFrame containing the pocket similarity between the pockets
        of KLIFS structures and a reference pocket.
        
        Parameters
        ----------
        reference_pocket: str
            The kinase pocket sequence the structures should be compared to.
        structures: pd.DataFrame
            A DataFrame containing KLIFS entries.
        
        Returns
        -------
        : pd.DataFrame
            The input DataFrame with a new 'pocket_similarity' column.
        """
        from ..modeling.alignment import sequence_similarity

        logging.debug("Calculating string similarity between KLIFS pockets ...")
        pocket_similarities = [
            sequence_similarity(structure_pocket, reference_pocket) for structure_pocket
            in structures["structure.pocket"]
        ]

        logging.debug("Adding pocket similarity to dataframe...")
        structures["pocket_similarity"] = pocket_similarities

        return structures

    def _get_design_unit(
        self,
        protein_structure: oechem.OEMolBase,
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
        protein_structure: oechem.OEMolBase
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

        if alternate_location == "-":
            alternate_location = None

        # generate unique design unit name
        design_unit_path = LocalFileStorage.featurizer_result(
            self.__class__.__name__,
            "_".join([
                structure_identifier,
                f"ligand{ligand_name}",
                f"chain{chain_id}",
                f"altloc{alternate_location}"
            ]),
            "oedu",
            self.cache_dir
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
        kinase_structure: oechem.OEMolBase,
        kinase_domain_sequence: Biosequence,
        chain_id: Union[str, None] = None
    ) -> oechem.OEMolBase:
        """
        Process a kinase domain according to UniProt.
        
        Parameters
        ----------
        kinase_structure: oechem.OEMolBase
            An OpenEye molecule holding the kinase structure to process.
        kinase_domain_sequence: Biosequence
            The kinase domain sequence with associated metadata.
        chain_id: str or None
            The chain of the kinase. Other chains will be deleted.
        
        Returns
        -------
        : oechem.OEMolBase
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

        logging.debug("Deleting loose protein segments after applying deletions ...")
        kinase_structure = delete_short_protein_segments(kinase_structure)

        logging.debug("Applying mutations to kinase domain ...")
        kinase_structure = apply_mutations(kinase_structure, kinase_domain_sequence)

        logging.debug("Deleting loose protein segments after applying mutations ...")
        kinase_structure = delete_short_protein_segments(kinase_structure)

        logging.debug("Renumbering residues ...")
        residue_numbers = self._get_kinase_residue_numbers(kinase_structure, kinase_domain_sequence)
        kinase_structure = renumber_structure(kinase_structure, residue_numbers)

        if self.loop_db:
            logging.debug("Applying insertions to kinase domain ...")
            kinase_structure = apply_insertions(kinase_structure, kinase_domain_sequence, self.loop_db)


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
        kinase_domain_structure: oechem.OEMolBase,
        kinase_domain_sequence: Biosequence
    ) -> List[int]:
        """
        Get the canonical residue numbers of a kinase domain structure.
        
        Parameters
        ----------
        kinase_domain_structure: oechem.OEMolBase
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
                # I don't this this will ever happen in the current implementation
                text = (
                    "Cannot generate residue IDs. The given protein structure contain residues "
                    "that are not part of the canoical sequence from UniProt."
                )
                logging.debug("Exception: " + text)
                raise NotImplementedError(text)
        return residue_numbers