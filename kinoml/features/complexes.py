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
from ..core.sequences import KinaseDomainAminoAcidSequence
from ..core.systems import ProteinLigandComplex


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

        logging.debug("Retrieving Featurizer results ...")
        protein_ligand_complex = self._get_featurizer_results(system, prepared_protein, prepared_solvent, docking_pose)

        return protein_ligand_complex

    @staticmethod
    def _interpret_system(system: ProteinLigandComplex) -> Tuple[oechem.OEGraphMol, oechem.OEGraphMol, Union[oegrid.OESkewGrid, None]]:
        """
        Interpret the given system components and retrieve OpenEye objects holding ligand, protein and electron density.
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
        from ..utils import FileDownloader

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

    def _get_design_unit(self, complex_structure: oechem.OEGraphMol, design_unit_identifier: str, electron_density: Union[oegrid.OESkewGrid, None]) -> oechem.OEDesignUnit:
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

        design_unit_path = LocalFileStorage.featurizer_result(self.__class__.__name__, f"{design_unit_identifier}_design_unit", "oedu")  # TODO: the file name needs to be unique
        if design_unit_path.is_file():
            logging.debug("Reading design unit from file ...")
            design_unit = oechem.OEDesignUnit()
            oechem.OEReadDesignUnit(str(design_unit_path), design_unit)
        else:
            logging.debug("Generating design unit ...")
            design_unit = prepare_complex(complex_structure, electron_density, self.loop_db)
            logging.debug("Writing design unit ...")
            oechem.OEWriteDesignUnit(str(design_unit_path), design_unit)

        return design_unit

    @staticmethod
    def _get_components(design_unit: oechem.OEDesignUnit) -> Tuple[oechem.OEGraphMol(), oechem.OEGraphMol(), oechem.OEGraphMol()]:
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

        logging.debug(f"Number of component atoms: Protein - {protein.NumAtoms()}, Solvent - {solvent.NumAtoms()}, Ligand - {ligand.NumAtoms()}.")
        return protein, solvent, ligand

    def _get_featurizer_results(self, system: ProteinLigandComplex, protein: oechem.OEGraphMol, solvent: oechem.OEGraphMol, docking_pose: oechem.OEGraphMol, other_pdb_header_info: Union[None, List[Tuple[str, str]]] = None) -> ProteinLigandComplex:
        """
        Get results from the Featurizer.
        Parameters
        ----------
        system: ProteinLigandComplex
            The system that was featurized.
        protein: oechem.OEGraphMol
            An OpenEye molecule holding the prepared protein.
        solvent: oechem.OEGraphMol
            An OpenEye molecule holding the prepared solvent.
        docking_pose: oechem.OEGraphMol
            An OpenEye molecule holding the docking pose.
        other_pdb_header_info: None or list of tuple of str
            Tuples with information that should be saved in the PDB header. Each tuple consists of two strings,
            i.e., the PDB header section (e.g. COMPND) and the respective information.
        Returns
        -------
        : ProteinLigandComplex
            A system holding the Featurizer results.
        """
        from ..modeling.OEModeling import remove_non_protein

        logging.debug("Assembling molecular components ...")
        protein_ligand_complex = self._assemble_complex(protein, solvent, docking_pose)
        solvated_protein = remove_non_protein(protein_ligand_complex, remove_water=False)

        logging.debug("Writing results ...")
        solvated_protein_path, ligand_path = self._write_results(system, solvated_protein, docking_pose, protein_ligand_complex, other_pdb_header_info=other_pdb_header_info)

        logging.debug("Generating new system components ...")
        file_protein = FileProtein(path=solvated_protein_path)
        file_ligand = FileLigand(path=ligand_path)
        protein_ligand_complex = ProteinLigandComplex(
            components=[file_protein, file_ligand]
        )

        return protein_ligand_complex

    @staticmethod
    def _assemble_complex(protein: oechem.OEGraphMol, solvent: oechem.OEGraphMol, ligand: oechem.OEGraphMol) -> oechem.OEGraphMol:
        """
        Assemble components of a solvated protein-ligand complex into a single OpenEye molecule. Water molecules will
        be checked for clashes with the ligand.
        Parameters
        ----------
        protein: oechem.OEGraphMol
            An OpenEye molecule holding the protein of interest.
        solvent: oechem.OEGraphMol
            An OpenEye molecule holding the solvent of interest.
        ligand: oechem.OEGraphMol
            An OpenEye molecule holding the ligand of interest.
        Returns
        -------
        protein_ligand_complex: oechem.OEGraphMol
            An OpenEye molecule holding protein, ligand and solvent.
        """
        from openeye import oechem

        from ..modeling.OEModeling import clashing_atoms, update_residue_identifiers, split_molecule_components

        protein_ligand_complex = oechem.OEGraphMol()

        logging.debug("Adding protein ...")
        oechem.OEAddMols(protein_ligand_complex, protein)

        logging.debug("Adding ligand ...")
        oechem.OEAddMols(protein_ligand_complex, ligand)

        logging.debug("Adding water molecules ...")
        water_molecules = split_molecule_components(solvent)  # converts solvent molecule into a list of water molecules
        for water_molecule in water_molecules:  # TODO: slow, move to cKDtrees
            if not clashing_atoms(ligand, water_molecule):
                oechem.OEAddMols(protein_ligand_complex, water_molecule)

        logging.debug("Updating hydrogen positions ...")
        oechem.OEPlaceHydrogens(protein_ligand_complex)

        logging.debug("Updating residue identifiers ...")
        protein_ligand_complex = update_residue_identifiers(protein_ligand_complex)

        return protein_ligand_complex

    def _write_results(self, system: ProteinLigandComplex, solvated_protein: oechem.OEGraphMol, ligand: oechem.OEGraphMol, protein_ligand_complex: oechem.OEGraphMol, other_pdb_header_info: Union[None, List[Tuple[str, str]]]) -> Tuple[str, str]:
        """
        Write the docking results from the Featurizer and retrieve the paths to protein and ligand.
        Parameters
        ----------
        system: ProteinLigandComplex
            The system to featurize.
        solvated_protein: oechem.OEGraphMol
            The OpenEye molecule holding the protein and solvent molecules not clashing with the docked ligand.
        ligand: oechem.OEGraphMol
            The OpenEye molecule holding the docked ligand.
        protein_ligand_complex: oechem.OEGraphMol
            The OpenEye molecule holding protein, solvent and ligand.
        other_pdb_header_info: None or list of tuple of str
            Tuples with information that should be saved in the PDB header. Each tuple consists of two strings,
            i.e., the PDB header section (e.g. COMPND) and the respective information.
        Returns
        -------
        : tuple of str and str
            Paths to prepared protein and docked ligand structure.
        """
        from ..modeling.OEModeling import write_molecules
        from ..utils import LocalFileStorage

        logging.debug("Writing protein ...")
        protein = self._update_pdb_header(solvated_protein, protein_name=system.protein.name, solvent_clashing_ligand_name=system.ligand.name, ligand_name="", other_pdb_header_info=other_pdb_header_info)
        protein_path = LocalFileStorage.featurizer_result(self.__class__.__name__, f"{system.protein.name}_{system.ligand.name}_protein", "pdb")
        write_molecules([protein], protein_path)

        logging.debug("Writing ligand ...")
        ligand_path = LocalFileStorage.featurizer_result(self.__class__.__name__, f"{system.protein.name}_{system.ligand.name}_ligand", "sdf")
        write_molecules([ligand], ligand_path)

        logging.debug("Writing protein ligand complex ...")
        protein_ligand_complex = self._update_pdb_header(protein_ligand_complex, protein_name=system.protein.name, solvent_clashing_ligand_name=system.ligand.name, ligand_name=system.ligand.name, other_pdb_header_info=other_pdb_header_info)
        complex_path = LocalFileStorage.featurizer_result(self.__class__.__name__, f"{system.protein.name}_{system.ligand.name}_complex", "pdb")
        write_molecules([protein_ligand_complex], complex_path)

        return protein_path, ligand_path

    def _update_pdb_header(self, structure: oechem.OEGraphMol, protein_name: str, solvent_clashing_ligand_name: str, ligand_name: str, other_pdb_header_info: Union[None, List[Tuple[str, str]]]) -> oechem.OEGraphMol:
        """
        Stores information about Featurizer, protein, solvent and ligand in the PDB header COMPND section in the
        given OpenEye molecule.
        Parameters
        ----------
        structure: oechem.OEGraphMol
            An OpenEye molecule.
        protein_name: str
            The name of the protein.
        solvent_clashing_ligand_name: str
            The name of the ligand that was used to remove clashing water molecules.
        ligand_name: str
            The name of the ligand.
        other_pdb_header_info: None or list of tuple of str
            Tuples with information that should be saved in the PDB header. Each tuple consists of two strings,
            i.e., the PDB header section (e.g. COMPND) and the respective information.
        Returns
        -------
        : oechem.OEGraphMol
            The OpenEye molecule containing the update PDB header.
        """
        from openeye import oechem

        oechem.OEClearPDBData(structure)
        oechem.OESetPDBData(structure, "COMPND", f"\tFeaturizer: {self.__class__.__name__}")
        oechem.OEAddPDBData(structure, "COMPND", f"\tProtein: {protein_name}")
        oechem.OEAddPDBData(structure, "COMPND", f"\tSolvent: Removed water clashing with {solvent_clashing_ligand_name}")
        oechem.OEAddPDBData(structure, "COMPND", f"\tLigand: {ligand_name}")
        if other_pdb_header_info is not None:
            for section, information in other_pdb_header_info:
                oechem.OEAddPDBData(structure, section, information)

        return structure


class OEKLIFSKinaseHybridDockingFeaturizer(OEHybridDockingFeaturizer):
    """
    Given a System with exactly one kinase and one ligand,
    dock the ligand in the designated binding pocket.

    We assume that the system contains a BaseProtein with a
    'klifs_kinase_id' attribute and a SmilesLigand.

    Parameters
    ----------
    loop_db: str
        The path to the loop database used by OESpruce to model missing loops.
    shape_overlay: bool
        If a shape overlay should be performed for selecting a ligand template
        in the hybrid docking protocol. Otherwise fingerprint similarity will
        be used.
    """

    import pandas as pd
    from openeye import oechem, oegrid

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
        import klifs_utils
        from openeye import oechem

        from ..docking.OEDocking import create_hybrid_receptor, hybrid_docking
        from ..modeling.OEModeling import compare_molecules, read_smiles
        from ..utils import LocalFileStorage

        if not hasattr(system.protein, "klifs_kinase_id"):
            raise NotImplementedError(f"{self.__class__.__name__} requires a system with a protein having a 'klifs_kinase_id' attribute.")

        if not hasattr(system.ligand, "smiles"):
            raise NotImplementedError(f"{self.__class__.__name__} requires a system with a ligand having a 'smiles' attribute.")

        logging.debug("Retrieving kinase details from KLIFS ...")
        kinase_details = klifs_utils.remote.kinases.kinases_from_kinase_ids(
            [system.protein.klifs_kinase_id]
        ).iloc[0]

        logging.debug("Searching ligand template ...")  # TODO: naming problem with co-crystallized ligand in hybrid docking, see above
        ligand_template = self._select_ligand_template(
            system.protein.klifs_kinase_id, read_smiles(system.ligand.smiles), self.shape_overlay
        )
        logging.debug(f"Selected {ligand_template.pdb} as ligand template ...")

        logging.debug("Searching kinase template ...")
        if ligand_template.kinase_ID == system.protein.klifs_kinase_id:
            protein_template = ligand_template
        else:
            protein_template = self._select_protein_template(
                system.protein.klifs_kinase_id,
                ligand_template.DFG,
                ligand_template.aC_helix,
            )
        logging.debug(f"Selected {protein_template.pdb} as kinase template ...")

        logging.debug(f"Adding attributes to BaseProtein ...")  # TODO: bad idea in a library
        system.protein.pdb_id = protein_template.pdb
        system.protein.path = LocalFileStorage.rcsb_structure_pdb(protein_template.pdb)
        system.protein.electron_density_path = LocalFileStorage.rcsb_electron_density_mtz(
            protein_template.pdb
        )

        logging.debug(f"Interpreting system ...")
        ligand, protein, electron_density = self._interpret_system(system)

        logging.debug(f"Preparing kinase template structure of {protein_template.pdb} ...")
        ligand_name = None
        if protein_template.ligand != 0:
            ligand_name = str(protein_template.ligand)
        design_unit = self._get_design_unit(protein, protein_template.pdb, electron_density, ligand_name)

        logging.debug("Extracting components ...")
        prepared_kinase, prepared_solvent = self._get_components(design_unit)[:2]

        logging.debug("Processing kinase domain ...")
        processed_kinase_domain = self._process_kinase_domain(prepared_kinase, kinase_details.uniprot)

        logging.debug(f"Preparing ligand template structure of {ligand_template.pdb} ...")
        prepared_ligand_template = self._prepare_ligand_template(ligand_template, processed_kinase_domain)

        logging.debug("Checking for co-crystallized ligand ...")
        if (
            compare_molecules(ligand, prepared_ligand_template)
            and ligand_template.pdb == protein_template.pdb
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

        logging.debug("Retrieving Featurizer results ...")
        kinase_ligand_complex = self._get_featurizer_results(system, processed_kinase_domain, prepared_solvent, docking_pose, other_pdb_header_info=[("COMPND", f"\tKinase template: {protein_template.pdb}"), ("COMPND", f"\tLigand template: {ligand_template.pdb}")])

        return kinase_ligand_complex  # TODO: MDAnalysis objects

    def _select_ligand_template(self, klifs_kinase_id: int, ligand: oechem.OEGraphMol, shape_overlay: bool) -> pd.Series:
        """
        Select a kinase in complex with a ligand from KLIFS holding a ligand similar to the given SMILES and bound to
        a kinase similar to the kinase of interest.
        Parameters
        ----------
        klifs_kinase_id: int
            KLIFS kinase identifier.
        ligand: oechem.OEGraphMol
            An OpenEye molecule holding the ligand that should be docked.
        shape_overlay: bool
            If shape should be considered for identifying similar ligands.
        Returns
        -------
        : pd.Series
            Details about selected kinase and co-crystallized ligand.
        """
        import klifs_utils

        logging.debug("Retrieve kinase information from KLIFS ...")
        kinase_details = klifs_utils.remote.kinases.kinases_from_kinase_ids([klifs_kinase_id]).iloc[0]

        logging.debug("Retrieve kinase structures from KLIFS and filter for orthosteric ligands ...")
        structures = self._get_available_ligand_templates()

        logging.debug("Storing SMILES in structures dataframe ...")
        structures = self._add_smiles_column(structures)

        logging.debug("Searching for identical co-crystallized ligands ...")
        identical_ligands = self._get_identical_ligand_indices(ligand, structures.smiles)  # TODO: Takes surprisingly long

        if len(identical_ligands) > 0:
            logging.debug("Found identical co-crystallized ligands ...")
            structures = structures.iloc[identical_ligands]
            logging.debug("Searching for matching KLIFS kinase id ...")
            if structures.kinase_ID.isin([kinase_details.kinase_ID]).any():
                logging.debug("Found matching KLIFS kinase id ...")
                structures = structures[
                    structures.kinase_ID.isin([kinase_details.kinase_ID])
                ]
        else:
            if shape_overlay:
                logging.debug("Filtering for most similar ligands according to their shape overlay ...")
                structures = self._filter_for_similar_ligands_3d(ligand, structures)
            else:
                logging.debug("Filtering for most similar ligands according to their fingerprints ...")
                structures = self._filter_for_similar_ligands_2d(ligand, structures)

        logging.debug("Filtering for most similar kinase pockets ...")
        structures = self._filter_for_similar_kinase_pockets(kinase_details.pocket, structures)

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
        import klifs_utils

        logging.debug("Retrieving available KLIFS entries ...")
        kinase_ids = klifs_utils.remote.kinases.kinase_names().kinase_ID.to_list()
        structures = klifs_utils.remote.structures.structures_from_kinase_ids(
            kinase_ids
        )

        logging.debug("Filtering KLIFS entries ...")
        structures = structures[structures["ligand"] != 0]  # orthosteric ligand
        structures = structures.groupby("pdb").filter(
            lambda x: len(set(x["ligand"])) == 1
        )  # single orthosteric ligand
        structures = structures[
            structures.allosteric_ligand == 0
            ]  # no allosteric ligand

        logging.debug("Sorting KLIFS entries by quality score...")
        # keep entry with highest quality score (alt 'A' preferred over alt 'B', chain 'A' preferred over 'B')
        structures = structures.sort_values(
            by=["alt", "chain", "quality_score"], ascending=[True, True, False]
        )

        logging.debug("Filtering for highest quality KLIFS entry per PDB code ...")
        structures = structures.groupby("pdb").head(1)

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
            smiles_from_pdb(set(structures.ligand) - set(pdb_to_smiles.keys()))
        )

        logging.debug("Saving local PDB SMILES dictionary ...")
        with open(LocalFileStorage.pdb_smiles_json(), "w") as wf:
            json.dump(pdb_to_smiles, wf)

        logging.debug("Adding SMILES to DataFrame ...")
        smiles_column = []
        for ligand_id in structures.ligand:
            if ligand_id in pdb_to_smiles.keys():
                smiles_column.append(pdb_to_smiles[ligand_id])
            else:
                smiles_column.append(None)
        structures["smiles"] = smiles_column

        logging.debug("Removing structures without a SMILES ...")
        structures = structures[structures.smiles.notnull()]

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
        complex_ligands = [get_klifs_ligand(structure_id) for structure_id in structures.structure_ID]

        logging.debug("Generating reasonable conformations of ligand of interest ...")
        conformations_ensemble = generate_reasonable_conformations(ligand)

        logging.debug("Overlaying molecules ...")
        overlay_scores = []
        for conformations in conformations_ensemble:
            overlay_scores += [[i, overlay_molecules(complex_ligand, conformations, False)] for i, complex_ligand in enumerate(complex_ligands)]

        # if maximal score is 1.73, threshold is set to 1.53
        overlay_score_threshold = max([score[1] for score in overlay_scores]) - 0.2

        logging.debug("Picking structures with most similar ligands ...")
        structures = structures.iloc[[score[0] for score in overlay_scores if score[1] >= overlay_score_threshold ]]

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
        rdkit_molecules = [Chem.MolFromSmiles(smiles) for smiles in structures.smiles]

        logging.debug("Adding RDKit molecules to dataframe...")
        structures["rdkit_molecules"] = rdkit_molecules

        logging.debug("Removing KLIFS entries without valid RDKit molecule ...")
        structures = structures[structures.rdkit_molecules.notnull()]

        logging.debug("Adding Feature Morgan fingerprint to dataframe...")
        pd.options.mode.chained_assignment = None  # otherwise next line would raise a warning
        structures["rdkit_fingerprint"] = [AllChem.GetMorganFingerprint(rdkit_molecule, 2, useFeatures=True) for rdkit_molecule in structures.rdkit_molecules]

        logging.debug("Generating Feature Morgan fingerprint of ligand ...")
        ligand_fingerprint = AllChem.GetMorganFingerprint(ligand, 2, useFeatures=True)

        logging.debug("Calculating dice similarity between fingerprints ...")
        fingerprint_similarities = [[i, DataStructs.DiceSimilarity(ligand_fingerprint, fingerprint)] for i, fingerprint in enumerate(structures.rdkit_fingerprint)]

        # if maximal score is 0.87, threshold is set to 0.77
        fingerprint_similarity_threshold = max([similarity[1] for similarity in fingerprint_similarities]) - 0.1

        logging.debug("Picking structures with most similar ligands ...")
        structures = structures.iloc[[similarity[0] for similarity in fingerprint_similarities if similarity[1] >= fingerprint_similarity_threshold]]

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
        from ..modeling.OEModeling import string_similarity

        logging.debug("Calculating string similarity between KLIFS pockets ...")
        pocket_similarities = [string_similarity(structure_pocket, reference_pocket) for structure_pocket in structures.pocket]

        logging.debug("Adding pocket similarity to dataframe...")
        structures["pocket_similarity"] = pocket_similarities

        # if maximal score is 0.87, threshold is set to 0.77
        pocket_similarity_threshold = max(pocket_similarities) - 0.1

        logging.debug("Picking structures with most similar kinase pockets ...")
        structures = structures[structures.pocket_similarity >= pocket_similarity_threshold]

        return structures

    @staticmethod
    def _select_protein_template(
        klifs_kinase_id: int,
        dfg: Union[str, None] = None,
        alpha_c_helix: Union[str, None] = None,
    ):
        """
        Select a kinase structure from KLIFS holding a kinase structure similar to the kinase of interest and with the
        specified conformation.
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
        import klifs_utils

        logging.debug("Retrieve kinase information from KLIFS ...")
        kinase_details = klifs_utils.remote.kinases.kinases_from_kinase_ids([klifs_kinase_id]).iloc[0]

        logging.debug("Retrieve kinase structures from KLIFS matching the kinase of interest ...")
        structures = klifs_utils.remote.structures.structures_from_kinase_ids([kinase_details.kinase_ID])

        logging.debug("Filtering KLIFS structures to match given kinase conformation ...")
        if dfg is not None:
            structures = structures[structures.DFG == dfg]
        if alpha_c_helix is not None:
            structures = structures[structures.aC_helix == alpha_c_helix]

        if len(structures) == 0:
            raise NotImplementedError("Would need homology modeling capability.")
        else:
            logging.debug("Picking structure with highest KLIFS quality score ...")
            structures = structures.sort_values(
                by=["alt", "chain", "quality_score"], ascending=[True, True, False]
            )
            protein_structure = structures.iloc[0]

        return protein_structure

    def _get_design_unit(self, protein_structure: oechem.OEGraphMol, design_unit_identifier: str, electron_density: Union[oegrid.OESkewGrid, None] = None, ligand_name: Union[str, None] = None) -> oechem.OEDesignUnit:
        """
        Get an OpenEye design unit from a protein ligand complex.
        Parameters
        ----------
        protein_structure: oechem.OEGraphMol
            An OpenEye molecule holding the protein that might be in complex with a ligand.
        design_unit_identifier: str
            A unique identifier describing the design unit.
        electron_density: oegrid.OESkewGrid or None
            An OpenEye grid holding the electron density of the protein ligand complex.
        ligand_name: str or None
            Residue name of the ligand in complex with the protein structure.
        Returns
        -------
        : oechem.OEDesignUnit
            The design unit.
        """
        from openeye import oechem

        from ..modeling.OEModeling import prepare_complex, prepare_protein
        from ..utils import LocalFileStorage

        design_unit_path = LocalFileStorage.featurizer_result(self.__class__.__name__, f"{design_unit_identifier}_design_unit", "oedu")
        if design_unit_path.is_file():
            logging.debug("Reading design unit from file ...")
            design_unit = oechem.OEDesignUnit()
            oechem.OEReadDesignUnit(str(design_unit_path), design_unit)
        else:
            logging.debug("Generating design unit ...")
            if ligand_name is None:
                design_unit = prepare_protein(protein_structure, self.loop_db, cap_termini=False)
            else:
                design_unit = prepare_complex(protein_structure, electron_density, self.loop_db, ligand_name=ligand_name, cap_termini=False)
            logging.debug("Writing design unit ...")
            oechem.OEWriteDesignUnit(str(design_unit_path), design_unit)

        return design_unit

    def _process_kinase_domain(self, kinase_structure: oechem.OEGraphMol, uniprot_id: str) -> oechem.OEGraphMol:
        """
        Process a kinase domain according to UniProt.
        Parameters
        ----------
        kinase_structure: oechem.OEGraphMol
            An OpenEye molecule holding the kinase structure to process.
        uniprot_id: str
            The UniProt identifier.
        Returns
        -------
        : oechem.OEGraphMol
            An OpenEye molecule holding the processed kinase structure.
        """
        from openeye import oechem

        from ..core.sequences import KinaseDomainAminoAcidSequence
        from ..modeling.OEModeling import mutate_structure, renumber_structure, prepare_protein

        logging.debug(f"Retrieving kinase domain sequence details for UniProt entry {uniprot_id} ...")
        kinase_domain_sequence = KinaseDomainAminoAcidSequence.from_uniprot(uniprot_id)

        logging.debug(f"Mutating kinase domain ...")
        mutated_structure = mutate_structure(kinase_structure, kinase_domain_sequence)

        logging.debug(f"Renumbering residues ...")
        residue_numbers = self._get_kinase_residue_numbers(mutated_structure, kinase_domain_sequence)
        renumbered_structure = renumber_structure(mutated_structure, residue_numbers)

        logging.debug(f"Capping kinase domain ...")
        real_termini = []
        if kinase_domain_sequence.metadata["true_N_terminus"]:
            if kinase_domain_sequence.metadata["begin"] == residue_numbers[0]:
                real_termini.append(residue_numbers[0])
        if kinase_domain_sequence.metadata["true_C_terminus"]:
            if kinase_domain_sequence.metadata["end"] == residue_numbers[-1]:
                real_termini.append(residue_numbers[-1])
        if len(real_termini) == 0:
            real_termini = None
        design_unit = prepare_protein(renumbered_structure, cap_termini=True, real_termini=real_termini)

        logging.debug("Extracting protein ...")
        processed_kinase_domain = oechem.OEGraphMol()
        design_unit.GetProtein(processed_kinase_domain)

        return processed_kinase_domain

    def _prepare_ligand_template(self, ligand_template: pd.Series, kinase_domain: oechem.OEGraphMol) -> oechem.OEGraphMol:
        """
        Prepare a PDB structure containing the ligand template of interest.
        Parameters
        ----------
        ligand_template: pd.Series
            A data series containing entries 'pdb', 'chain', 'ligand' and 'alt'.
        kinase_domain: oechem.OEGraphMol
            An OpenEye molecule holding the kinase domain the ligand template structure should be superposed to.
        Returns
        -------
        : oechem.OEGraphMol
            An OpenEye molecule holding the prepared ligand structure.
        """
        from openeye import oechem
        from ..modeling.OEModeling import read_molecules, select_chain, select_altloc, remove_non_protein, superpose_proteins
        from ..utils import FileDownloader

        logging.debug("Interpreting structure ...")
        ligand_template_structure = PDBProtein(ligand_template.pdb)
        if not ligand_template_structure.path.is_file():
            logging.debug(
                f"Downloading PDB entry {ligand_template.pdb} ..."
            )
            FileDownloader.rcsb_structure_pdb(ligand_template.pdb)
        logging.debug("Reading structure ...")
        ligand_template_structure = read_molecules(ligand_template_structure.path)[0]

        logging.debug("Selecting chain ...")
        ligand_template_structure = select_chain(ligand_template_structure, ligand_template.chain)

        logging.debug("Selecting alternate location ...")
        ligand_template_structure = select_altloc(ligand_template_structure, ligand_template.alt)

        logging.debug("Removing everything but protein, water and ligand of interest ...")
        ligand_template_structure = remove_non_protein(ligand_template_structure, exceptions=[ligand_template.ligand], remove_water=False)

        logging.debug("Superposing structure on kinase domain ...")
        ligand_template_structure = superpose_proteins(kinase_domain, ligand_template_structure)

        logging.debug("Adding hydrogens ...")
        oechem.OEPlaceHydrogens(ligand_template_structure)
        split_options = oechem.OESplitMolComplexOptions()

        logging.debug("Extracting ligand ...")
        ligand_template_structure = list(oechem.OEGetMolComplexComponents(ligand_template_structure, split_options, split_options.GetLigandFilter()))[0]

        return ligand_template_structure

    @staticmethod
    def _get_kinase_residue_numbers(kinase_domain_structure: oechem.OEGraphMol, canonical_kinase_domain_sequence: KinaseDomainAminoAcidSequence) -> List[int]:
        """
        Get the canonical residue numbers of a kinase domain structure.
        Parameters
        ----------
        kinase_domain_structure: oechem.OEGraphMol
            The kinase domain structure.
        canonical_kinase_domain_sequence: KinaseDomainAminoAcidSequence
            The canonical kinase domain sequence.
        Returns
        -------
        residue_number: list of int
            A list of canonical residue numbers in the same order as the residues in the given kinase domain structure.
        """
        from Bio import pairwise2
        from kinoml.modeling.OEModeling import get_sequence

        logging.debug("Getting sequence of given kinase domain ...")
        target_sequence = get_sequence(kinase_domain_structure)

        logging.debug("Aligning sequences ...")
        template_sequence, target_sequence = pairwise2.align.globalxs(
            canonical_kinase_domain_sequence, target_sequence, -10, 0
        )[0][:2]
        logging.debug(f"Template sequence:\n{template_sequence}")
        logging.debug(f"Target sequence:\n{target_sequence}")

        logging.debug("Generating residue numbers ...")
        residue_numbers = []
        residue_number = canonical_kinase_domain_sequence.metadata["begin"]
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
