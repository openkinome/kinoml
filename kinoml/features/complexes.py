"""
Featurizers that can only get applied to ProteinLigandComplexes or
subclasses thereof
"""
from __future__ import annotations
from functools import lru_cache
import logging
from typing import Union

from .core import BaseFeaturizer
from ..core.ligands import FileLigand, SmilesLigand
from ..core.proteins import FileProtein, PDBProtein
from ..core.sequences import KinaseDomainAminoAcidSequence
from ..core.systems import ProteinLigandComplex


class OpenEyesHybridDockingFeaturizer(BaseFeaturizer):

    """
    Given a System with exactly one protein and one ligand,
    dock the ligand in the designated binding pocket.

    We assume that a smiles and file-based System object will be passed;
    this means we will have a System.components with FileProtein and
    FileLigand or SmilesLigand. The file itself could be a URL.
    """

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
        from appdirs import user_cache_dir
        from openeye import oechem

        from ..docking.OpenEyeDocking import create_hybrid_receptor, hybrid_docking
        from ..modeling.OpenEyeModeling import prepare_complex, write_molecules

        ligand, smiles, protein, electron_density = self.interpret_system(system)

        design_unit = prepare_complex(protein, electron_density, self.loop_db)
        prepared_protein = oechem.OEGraphMol()
        prepared_ligand = oechem.OEGraphMol()
        design_unit.GetProtein(prepared_protein)
        design_unit.GetLigand(prepared_ligand)
        hybrid_receptor = create_hybrid_receptor(prepared_protein, prepared_ligand)
        docking_poses = hybrid_docking(hybrid_receptor, [ligand])

        # TODO: where to store data
        protein_path = f"{user_cache_dir()}/{system.protein.name}.pdb"  # mmcif writing not supported by openeye
        write_molecules([prepared_protein], protein_path)
        file_protein = FileProtein(path=protein_path)

        ligand_path = (
            f"{user_cache_dir()}/{system.protein.name}_{system.ligand.name}.sdf"
        )
        write_molecules(docking_poses, ligand_path)
        file_ligand = FileLigand(path=ligand_path)
        protein_ligand_complex = ProteinLigandComplex(
            components=[file_protein, file_ligand]
        )
        return protein_ligand_complex

    @staticmethod
    def interpret_system(system):
        from openeye import oechem
        from ..modeling.OpenEyeModeling import read_smiles, read_molecules, read_electron_density
        from ..utils import FileDownloader

        # ligand
        logging.debug("Interpreting ligand ...")
        if isinstance(system.ligand, SmilesLigand):
            smiles = system.ligand.smiles
            logging.debug("Loading ligand from SMILES string ...")
            ligand = read_smiles(system.ligand.smiles)
        else:
            logging.debug(f"Loading ligand from {system.ligand.path} ...")
            ligand = read_molecules(system.ligand.path)[0]
            logging.debug("Converting ligand to SMILES string ...")
            smiles = oechem.OEMolToSmiles(ligand)

        # protein
        logging.debug("Interpreting protein ...")
        if hasattr(system.protein, 'pdb_id'):
            if not system.protein.path.is_file():
                logging.debug(f"Downloading protein structure {system.protein.pdb_id} from PDB ...")
                FileDownloader.rcsb_structure_pdb(system.protein.pdb_id)
        logging.debug(f"Reading protein structure from {system.protein.path} ...")
        protein = read_molecules(system.protein.path)[0]

        # electron density
        logging.debug("Interpreting electron density ...")
        electron_density = None
        # TODO: Kills Kernel for some reason
        #if system.protein.electron_density_path is not None:
        #    if hasattr(system.protein, 'pdb_id'):
        #        if not system.protein.electron_density_path.is_file():
        #            logging.debug(f"Downloading electron density for structure {system.protein.pdb_id} from PDB ...")
        #            FileDownloader.rcsb_electron_density_mtz(system.protein.pdb_id)
        #    logging.debug(f"Reading electron density from {system.protein.electron_density_path} ...")
        #    electron_density = read_electron_density(system.protein.electron_density_path)
        logging.debug("Returning system components...")
        return ligand, smiles, protein, electron_density


class OpenEyesKLIFSKinaseHybridDockingFeaturizer(OpenEyesHybridDockingFeaturizer):
    """
    Given a System with exactly one kinase and one ligand,
    dock the ligand in the designated binding pocket.

    We assume that the system contains a BaseProtein with a
    'klifs_kinase_id' attribute and a SmilesLigand.
    """
    import pandas as pd

    def __init__(self, loop_db: Union[str, None] = None, shape: bool = False, debug: bool = False):
        super().__init__(loop_db)
        self.loop_db = loop_db
        self.shape = shape
        if debug:
            logging.basicConfig(level=logging.DEBUG)

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
        from ..docking.OpenEyeDocking import create_hybrid_receptor, hybrid_docking
        from ..modeling.OpenEyeModeling import select_chain, select_altloc, remove_non_protein, prepare_complex, prepare_protein, mutate_structure, renumber_structure, write_molecules, read_molecules, superpose_proteins, compare_molecules, update_residue_identifiers, clashing_atoms
        from ..utils import LocalFileStorage, FileDownloader

        if not hasattr(system.protein, 'klifs_kinase_id'):
            print("The protein misses a 'klifs_kinase_id' attribute.")
            raise NotImplementedError

        if not isinstance(system.ligand, SmilesLigand):
            print("This featurizer needs a system with at SmilesLigand.")
            raise NotImplementedError

        kinase_details = klifs_utils.remote.kinases.kinases_from_kinase_ids(
            [system.protein.klifs_kinase_id]
        ).iloc[0]

        logging.debug("Searching ligand template ...")
        # select structure for ligand modeling
        ligand_template = self.select_ligand_template(system.protein.klifs_kinase_id, system.ligand.smiles, self.shape)
        logging.debug(f"Selected {ligand_template.pdb} as ligand template ...")

        logging.debug("Searching kinase template ...")
        # select structure for kinase modeling
        if ligand_template.kinase_ID == system.protein.klifs_kinase_id:
            protein_template = ligand_template
        else:
            protein_template = self.select_protein_template(system.protein.klifs_kinase_id, ligand_template.DFG, ligand_template.aC_helix)
        logging.debug(f"Selected {protein_template.pdb} as kinase template ...")

        logging.debug(f"Adding attributes to BaseProtein ...")
        system.protein.pdb_id = protein_template.pdb
        system.protein.path = LocalFileStorage.rcsb_structure_pdb(protein_template.pdb)
        system.protein.electron_density_path = LocalFileStorage.rcsb_electron_density_mtz(protein_template.pdb)

        logging.debug(f"Interpreting system ...")
        ligand, smiles, protein, electron_density = self.interpret_system(system)

        logging.debug(f"Preparing kinase domain of {protein_template.pdb} ...")
        kinase_domain_path = LocalFileStorage.rcsb_kinase_domain_pdb(protein_template.pdb)
        if not kinase_domain_path.is_file():
            logging.debug(f"Generating design unit ...")
            design_unit = prepare_complex(protein, electron_density, self.loop_db, ligand_name=str(protein_template.ligand), cap_termini=False)
            logging.debug(f"Extracting protein ...")
            protein = oechem.OEGraphMol()
            design_unit.GetProtein(protein)
            logging.debug(f"Extracting solvent ...")
            solvent = oechem.OEGraphMol()
            design_unit.GetSolvent(solvent)
            logging.debug(f"Retrieving kinase domain sequence for {kinase_details.uniprot} ...")
            kinase_domain_sequence = KinaseDomainAminoAcidSequence.from_uniprot(kinase_details.uniprot)
            logging.debug(f"Mutating kinase domain ...")
            mutated_structure = mutate_structure(protein, kinase_domain_sequence)
            residue_numbers = self.get_kinase_residue_numbers(mutated_structure, kinase_domain_sequence)
            logging.debug(f"Renumbering residues ...")
            renumbered_structure = renumber_structure(mutated_structure, residue_numbers)
            logging.debug(f"Adding solvent to standardized kinase domain ...")
            _a, _b = oechem.OEAddMols(renumbered_structure, solvent)
            real_termini = []
            if kinase_domain_sequence.metadata["true_N_terminus"]:
                if kinase_domain_sequence.metadata["begin"] == residue_numbers[0]:
                    real_termini.append(residue_numbers[0])
            if kinase_domain_sequence.metadata["true_C_terminus"]:
                if kinase_domain_sequence.metadata["end"] == residue_numbers[-1]:
                    real_termini.append(residue_numbers[-1])
            if len(real_termini) == 0:
                real_termini = None
            logging.debug(f"Capping kinase domain ...")
            design_unit = prepare_protein(renumbered_structure, cap_termini=True, real_termini=real_termini)
            solvated_kinase_domain = oechem.OEGraphMol()
            components = oechem.OEDesignUnitComponents_Protein | oechem.OEDesignUnitComponents_Solvent
            design_unit.GetComponents(solvated_kinase_domain, components)
            solvated_kinase_domain = update_residue_identifiers(solvated_kinase_domain)
            logging.debug(f"Writing kinase domain to {kinase_domain_path}...")
            write_molecules([solvated_kinase_domain], kinase_domain_path)
        else:
            logging.debug(f"Reading kinase domain from {kinase_domain_path} ...")
            solvated_kinase_domain = read_molecules(kinase_domain_path)[0]

        logging.debug(f"Preparing ligand of {ligand_template.pdb} ...")
        ligand_template_structure = PDBProtein(ligand_template.pdb)
        logging.debug("Interpreting protein ...")
        if not ligand_template_structure.path.is_file():
            logging.debug(f"Downloading protein structure {ligand_template_structure.pdb_id} from PDB ...")
            FileDownloader.rcsb_structure_pdb(ligand_template_structure.pdb_id)
        ligand_template_structure = read_molecules(ligand_template_structure.path)[0]
        ligand_template_structure = select_chain(ligand_template_structure, ligand_template.chain)
        ligand_template_structure = select_altloc(ligand_template_structure, ligand_template.alt)
        ligand_template_structure = remove_non_protein(ligand_template_structure, exceptions=[ligand_template.ligand], remove_water=False)
        ligand_template_structure = superpose_proteins(solvated_kinase_domain, ligand_template_structure)
        oechem.OEPlaceHydrogens(ligand_template_structure)
        split_options = oechem.OESplitMolComplexOptions()
        ligand_template_structure = list(oechem.OEGetMolComplexComponents(ligand_template_structure, split_options, split_options.GetLigandFilter()))[0]
        oechem.OEClearPDBData(ligand_template_structure)  # important for later merging with kinase domain

        if compare_molecules(ligand, ligand_template_structure) and ligand_template.pdb == protein_template.pdb:
            logging.debug(f"Found co-crystallized ligand ...")
            docking_pose = ligand_template_structure
        else:
            logging.debug(f"Docking ligand into kinase domain ...")
            hybrid_receptor = create_hybrid_receptor(solvated_kinase_domain, ligand_template_structure)
            docking_pose = hybrid_docking(hybrid_receptor, [ligand])[0]
            # generate residue information
            oechem.OEPerceiveResidues(docking_pose, oechem.OEPreserveResInfo_None)

        logging.debug("Writing docking pose ...")
        docking_pose_path = LocalFileStorage.DIRECTORY / f"rcsb_{protein_template.pdb}_kinase_domain_{system.ligand.name}.sdf"
        write_molecules([docking_pose], docking_pose_path)

        logging.debug("Assembling kinase ligand complex ...")
        split_options = oechem.OESplitMolComplexOptions()
        kinase_domain = list(oechem.OEGetMolComplexComponents(solvated_kinase_domain, split_options, split_options.GetProteinFilter()))[0]
        kinase_ligand_complex = oechem.OEGraphMol()
        logging.debug("Adding kinase domain ...")
        _a, _b = oechem.OEAddMols(kinase_ligand_complex, kinase_domain)
        logging.debug("Adding ligand ...")
        _a, _b = oechem.OEAddMols(kinase_ligand_complex, docking_pose)
        solvent = list(oechem.OEGetMolComplexComponents(solvated_kinase_domain, split_options, split_options.GetWaterFilter()))
        logging.debug("Adding water molecules ...")
        for water_molecule in solvent:
            if not clashing_atoms(docking_pose, water_molecule):
                _a, _b = oechem.OEAddMols(kinase_ligand_complex, water_molecule)
        oechem.OEPlaceHydrogens(kinase_ligand_complex)
        kinase_ligand_complex = update_residue_identifiers(kinase_ligand_complex)

        logging.debug("Writing kinase ligand complex ...")
        complex_path = LocalFileStorage.DIRECTORY / f"{system.protein.name}_{system.ligand.name}.pdb"
        write_molecules([kinase_ligand_complex], complex_path)

        file_protein = FileProtein(path=str(LocalFileStorage.rcsb_kinase_domain_pdb(protein_template.pdb)))
        file_ligand = FileLigand(path=str(docking_pose_path))

        kinase_ligand_complex = ProteinLigandComplex(components=[file_protein, file_ligand])

        return kinase_ligand_complex

    @staticmethod
    def select_ligand_template(
            klifs_kinase_id: int, smiles: str, shape: bool) -> pd.DataFrame:
        """
        Select a kinase in complex with a ligand from KLIFS holding a ligand similar to the given SMILES and bound to
        a kinase similar to the kinase of interest.
        Parameters
        ----------
        klifs_kinase_id: int
            KLIFS kinase identifier.
        smiles: str
            The molecule in smiles format.
        shape: bool
            If shape should be considered for identifying similar ligands.
        Returns
        -------
            : pd.Series
            Details about selected kinase and co-crystallized ligand.
        """
        import json

        import klifs_utils
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem, DataStructs

        from ..modeling.OpenEyeModeling import read_smiles, smiles_from_pdb, compare_molecules, get_klifs_ligand, generate_reasonable_conformations, overlay_molecules, string_similarity
        from ..utils import LocalFileStorage

        RDLogger.DisableLog("rdApp.*")

        # retrieve kinase information from KLIFS
        kinase_details = klifs_utils.remote.kinases.kinases_from_kinase_ids(
            [klifs_kinase_id]
        ).iloc[0]

        # retrieve kinase structures from KLIFS and filter for orthosteric ligands
        kinase_ids = klifs_utils.remote.kinases.kinase_names().kinase_ID.to_list()
        structures = klifs_utils.remote.structures.structures_from_kinase_ids(kinase_ids)
        structures = structures[structures["ligand"] != 0]  # orthosteric ligand
        structures = structures.groupby("pdb").filter(
            lambda x: len(set(x["ligand"])) == 1
        )  # single orthosteric ligand
        structures = structures[structures.allosteric_ligand == 0]  # no allosteric ligand
        # keep entry with highest quality score (alt 'A' preferred over alt 'B', chain 'A' preferred over 'B')
        structures = structures.sort_values(
            by=["alt", "chain", "quality_score"], ascending=[True, True, False]
        )
        structures = structures.groupby("pdb").head(1)

        # store smiles in structures dataframe
        if LocalFileStorage.pdb_smiles_json().is_file():
            with open(LocalFileStorage.pdb_smiles_json(), "r") as rf:
                pdb_to_smiles = json.load(rf)
        else:
            pdb_to_smiles = {}
        pdb_to_smiles.update(smiles_from_pdb(set(structures.ligand) - set(pdb_to_smiles.keys())))
        with open(LocalFileStorage.pdb_smiles_json(), "w") as wf:
            json.dump(pdb_to_smiles, wf)
        smiles_column = []
        for ligand_id in structures.ligand:
            if ligand_id in pdb_to_smiles.keys():
                smiles_column.append(pdb_to_smiles[ligand_id])
            else:
                smiles_column.append(None)
        structures["smiles"] = smiles_column
        structures = structures[
            structures.smiles.notnull()
        ]  # remove structures with missing smiles

        # try to find identical co-crystallized ligands
        ligand = read_smiles(smiles)
        identical_ligands = []
        for i, complex_ligand in enumerate(structures.smiles):
            if compare_molecules(ligand, read_smiles(complex_ligand)):
                identical_ligands.append(i)
        if len(identical_ligands) > 0:
            structures = structures.iloc[identical_ligands]
            # try to find the same kinase
            if structures.kinase_ID.isin([kinase_details.kinase_ID]).any():
                structures = structures[
                    structures.kinase_ID.isin([kinase_details.kinase_ID])
                ]
        else:
            if shape:
                # get resolved structure of orthosteric ligands
                complex_ligands = [
                    get_klifs_ligand(structure_id)
                    for structure_id in structures.structure_ID
                ]

                # get reasonable conformations of ligand of interest
                conformations_ensemble = generate_reasonable_conformations(ligand)

                # overlay and score
                overlay_scores = []
                for conformations in conformations_ensemble:
                    overlay_scores += [
                        [i, overlay_molecules(complex_ligand, conformations, False)]
                        for i, complex_ligand in enumerate(complex_ligands)
                    ]
                overlay_score_threshold = max([score[1] for score in overlay_scores]) - 0.2

                # pick structures with similar ligands
                structures = structures.iloc[
                    [
                        score[0]
                        for score in overlay_scores
                        if score[1] >= overlay_score_threshold
                    ]
                ]
            else:
                rdkit_molecules = [
                    Chem.MolFromSmiles(smiles) for smiles in structures.smiles
                ]
                structures["rdkit_molecules"] = rdkit_molecules
                structures = structures[structures.rdkit_molecules.notnull()]
                structures["rdkit_fingerprint"] = [
                    AllChem.GetMorganFingerprint(rdkit_molecule, 2, useFeatures=True)
                    for rdkit_molecule in structures.rdkit_molecules
                ]
                ligand_fingerprint = AllChem.GetMorganFingerprint(
                    Chem.MolFromSmiles(smiles), 2, useFeatures=True
                )
                fingerprint_similarities = [
                    [i, DataStructs.DiceSimilarity(ligand_fingerprint, fingerprint)]
                    for i, fingerprint in enumerate(structures.rdkit_fingerprint)
                ]
                fingerprint_similarity_threshold = (
                        max([similarity[1] for similarity in fingerprint_similarities]) - 0.1
                )
                structures = structures.iloc[
                    [
                        similarity[0]
                        for similarity in fingerprint_similarities
                        if similarity[1] >= fingerprint_similarity_threshold
                    ]
                ]

        # find most similar kinase pockets
        pocket_similarities = [
            string_similarity(structure_pocket, kinase_details.pocket)
            for structure_pocket in structures.pocket
        ]
        structures["pocket_similarity"] = pocket_similarities
        pocket_similarity_threshold = max(pocket_similarities) - 0.1
        structures = structures[structures.pocket_similarity >= pocket_similarity_threshold]
        structure_for_ligand = structures.iloc[0]

        return structure_for_ligand

    @staticmethod
    def select_protein_template(
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

        # retrieve kinase information from KLIFS
        kinase_details = klifs_utils.remote.kinases.kinases_from_kinase_ids(
            [klifs_kinase_id]
        ).iloc[0]

        # retrieve kinase structures from KLIFS and filter for orthosteric ligands
        structures = klifs_utils.remote.structures.structures_from_kinase_ids(
            [kinase_details.kinase_ID]
        )
        if dfg is not None:
            structures = structures[structures.DFG == dfg]
        if alpha_c_helix is not None:
            structures = structures[structures.aC_helix == alpha_c_helix]

        if len(structures) == 0:
            # TODO: integrate homology modeling
            raise NotImplementedError
        else:
            structures = structures.sort_values(
                by=["alt", "chain", "quality_score"], ascending=[True, True, False]
            )
            protein_structure = structures.iloc[0]

        return protein_structure

    @staticmethod
    def get_kinase_residue_numbers(kinase_domain_structure, canonical_kinase_domain_sequence):
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
        from kinoml.modeling.OpenEyeModeling import get_sequence

        target_sequence = get_sequence(kinase_domain_structure)
        template_sequence, target_sequence = pairwise2.align.globalxs(canonical_kinase_domain_sequence, target_sequence, -10, 0)[0][:2]
        logging.debug(f"Template sequence:\n{template_sequence}")
        logging.debug(f"Target sequence:\n{target_sequence}")
        residue_numbers = []
        residue_number = canonical_kinase_domain_sequence.metadata["begin"]
        for template_sequence_residue, target_sequence_residue in zip(template_sequence, target_sequence):
            if template_sequence_residue != "-":
                if target_sequence_residue != "-":
                    residue_numbers.append(residue_number)
                residue_number += 1
            else:
                # TODO: This situation occurs if the given protein contains sequence segments that are not part of the
                #       canonical kinase domain sequence from UniProt. Not sure if this will ever happen in the current
                #       implementation.
                raise NotImplementedError
        return residue_numbers
