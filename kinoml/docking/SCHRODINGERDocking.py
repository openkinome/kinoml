import logging
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile
from typing import List, Union


logger = logging.getLogger(__name__)


def run_glide(
        schrodinger_directory: Union[Path, str],
        input_file_mae: Union[Path, str],
        output_file_sdf: Union[Path, str],
        ligand_resname: str,
        mols_smiles: Union[List[str]],
        n_poses: int = 1,
        mols_names: Union[List[str], None] = None,
        shape_restrain: bool = True,
        macrocyles: bool = False,
        precision: str = "XP",
):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from ..modeling.MDAnalysisModeling import read_molecule
    from ..modeling.SCHRODINGERModeling import mae_to_pdb

    if precision not in ["HTVS", "SP", "XP"]:
        raise ValueError(
            f"Only 'HTVS', 'SP', 'XP' are allowed for precision, you provided {precision}!"
        )

    schrodinger_directory = Path(schrodinger_directory).resolve()
    input_file_mae = Path(input_file_mae).resolve()
    with NamedTemporaryFile(mode="w", suffix=".mae") as protein_file_mae, \
            NamedTemporaryFile(mode="w", suffix=".pdb") as ligand_file_pdb, \
            NamedTemporaryFile(mode="w", suffix=".sdf") as mols_file_sdf, \
            NamedTemporaryFile(mode="w", suffix=".in") as grid_input_file, \
            NamedTemporaryFile(mode="w", suffix=".in") as docking_input_file:

        logger.debug("Selecting and writing protein from MAE input file ...")
        subprocess.run([
            str(schrodinger_directory / "run"),
            "delete_atoms.py",
            str(input_file_mae),
            protein_file_mae.name,
            "-asl",
            '"not protein"'
        ])

        logger.debug("Converting MAE to PDB ...")
        input_file_pdb = input_file_mae.parent / (input_file_mae.stem + ".pdb")
        mae_to_pdb(schrodinger_directory, input_file_mae, input_file_pdb)

        logger.debug("Selecting and writing co-crystallized ligand ...")
        structure = read_molecule(input_file_mae)
        ligands = structure.select_atoms(f"resname {ligand_resname}").residues
        if len(ligands) == 0:
            logger.debug(
                f"Could not find ligand {ligand_resname} in structure {input_file_pdb}, "
                f"cannot proceed to docking!"
            )
            return
        ligands[0].atoms.write(ligand_file_pdb.name)  # write first residue (in case of multiple chains)

        logger.debug("Writing molecules to SDF ...")
        if not mols_names or len(mols_names) != len(mols_smiles):
            logger.debug("Creating molecule names ...")
            mols_names = [str(x) for x in range(1, len(mols_smiles) + 1)]
        sd_writer = Chem.SDWriter(mols_file_sdf.name)
        for smiles, name in zip(mols_smiles, mols_names):
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                logger.debug(f"Skipping molecule {name} with erroneous smiles ...")
                continue
            mol.SetProp("_Name", name)
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            sd_writer.write(mol)

        logger.debug("Writing input file for grid generation ...")
        grid_input_file.write(f"RECEP_FILE '{protein_file_mae.name}'\n")
        grid_input_file.write(f"REF_LIGAND_FILE '{ligand_file_pdb.name}'\n")
        grid_input_file.write("LIGAND_INDEX 1\n")
        grid_input_file.flush()

        logger.debug("Generating grid for docking ...")
        subprocess.run([
            str(schrodinger_directory / "glide"),
            grid_input_file.name,
            "-HOST",
            "localhost",
            "-WAIT",
            "-OVERWRITE"
        ])

        logger.debug("Writing input file for docking ...")
        docking_input_file.write(f"GRIDFILE '{grid_input_file.name}'\n")
        docking_input_file.write(f"LIGANDFILE '{mols_file_sdf.name}'\n")
        docking_input_file.write("POSE_OUTTYPE ligandlib_sd\n")
        docking_input_file.write(f"COMPRESS_POSES False\n")
        docking_input_file.write(f"POSES_PER_LIG {n_poses}\n")
        docking_input_file.write(f"PRECISION {precision}\n")
        if shape_restrain:
            docking_input_file.write(f"SHAPE_RESTRAIN True\n")
            docking_input_file.write(f"SHAPE_REF_LIGAND_FILE '{ligand_file_pdb.name}'\n")
        if macrocyles:
            docking_input_file.write(f"MACROCYCLE True\n")
        docking_input_file.flush()

        logger.debug("Running docking ...")
        subprocess.run([
            str(schrodinger_directory / "glide"),
            docking_input_file.name,
            "-HOST",
            "localhost",
            "-WAIT",
            "-OVERWRITE"
        ])

        logger.debug("Filtering poses for appropriate number ...")
        grid_input_file_path = Path(grid_input_file.name)
        sd_file_path = grid_input_file_path.parent / (grid_input_file_path.stem + "_lib.sdf")
        if not sd_file_path:
            logger.debug("No docking poses were generated during docking ...")
            return
        supplier = Chem.SDMolSupplier(sd_file_path)
        sd_writer = Chem.SDWriter(output_file_sdf)
        mol_counter_dict = {}
        for mol in supplier:
            # SDF from glide is sorted by docking score, but mols are in mixed order
            name = mol.GetProp("_Name")
            if name not in mol_counter_dict.keys():
                mol_counter_dict[name] = 0
            if mol_counter_dict[name] <= n_poses:
                sd_writer.write(mol)
                mol_counter_dict[name] += 1
        sd_file_path.unlink()  # manually delete file

    return
