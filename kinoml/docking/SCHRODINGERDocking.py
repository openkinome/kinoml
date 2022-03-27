import logging
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile
from typing import List, Union

from appdirs import user_cache_dir


logger = logging.getLogger(__name__)


def run_glide(
    schrodinger_directory: Union[Path, str],
    input_file_mae: Union[Path, str],
    output_file_sdf: Union[Path, str],
    mols_smiles: List[str],
    ligand_resname: Union[str, None],
    n_poses: int = 1,
    mols_names: Union[List[str], None] = None,
    shape_restrain: bool = True,
    macrocyles: bool = False,
    precision: str = "XP",
    cache_dir: Union[Path, str] = user_cache_dir(),
):
    """
    Run glide for protein ligand docking.

    Parameters
    ----------
    schrodinger_directory: Path or str
        The path to the directory of the Schrodinger installation.
    input_file_mae: Path or str
        The path to the input file in MAE format containing the protein structure to dock to and a
        co-crystallized ligand in the binding pocket of interest.
    output_file_sdf: Path or str
        The path to the output file of the generated in docking poses in SDF format.
    mols_smiles: list of str
        The molecules to dock as SMILES representation.
    ligand_resname: str or None
        The resname of the co-crystallized ligand, which will be used for pocket definition.
    mols_names: None or list of str, default=None
        The names of the molecules to dock. Will be used as molecule title in the SDF file. If
        None, names will be numbers (1,..,len(mols_smiles).
    n_poses: int, default=1
        Number of poses to generate per molecule.
    shape_restrain: bool, default=True
        If the co-crystallized ligand shell be used for shape restrained docking.
    macrocyles: bool, default=False
        Macrocycle conformations will be sampled with an appropriate algorithm. All non-
        macrocyclic molecules by detected by SCHRODINGER will be skipped.
    precision: str, default="XP"
        The docking precision to use ["HTVS", "SP", "XP"].
    cache_dir: Path or str, default=appdirs.user_cache_dir()
        Path to a directory for caching grids for docking.
    """
    import shutil

    from rdkit import Chem
    from rdkit.Chem import AllChem

    from ..utils import sha256_objects

    if precision not in ["HTVS", "SP", "XP"]:
        raise ValueError(
            f"Only 'HTVS', 'SP', 'XP' are allowed for precision, you provided {precision}!"
        )

    schrodinger_directory = Path(schrodinger_directory).resolve()
    input_file_mae = Path(input_file_mae).resolve()
    with NamedTemporaryFile(mode="w", suffix=".mae") as protein_file_mae, NamedTemporaryFile(
        mode="w", suffix=".mae"
    ) as ligand_file_mae, NamedTemporaryFile(
        mode="w", suffix=".mae"
    ) as protein_ligand_file_mae, NamedTemporaryFile(
        mode="w", suffix=".sdf"
    ) as mols_file_sdf, NamedTemporaryFile(
        mode="w", suffix=".in"
    ) as grid_input_file, NamedTemporaryFile(
        mode="w", suffix=".in"
    ) as docking_input_file:

        logger.debug("Selecting and writing protein from MAE input file ...")
        subprocess.run(
            [
                str(schrodinger_directory / "run"),
                "delete_atoms.py",
                str(input_file_mae),
                protein_file_mae.name,
                "-asl",
                "not protein",
            ]
        )

        with NamedTemporaryFile(mode="w", suffix=".mae") as ligand_file_raw_mae:
            logger.debug("Selecting and writing ligand from MAE input file ...")
            subprocess.run(  # first everything that could be ligand
                [
                    str(schrodinger_directory / "run"),
                    "delete_atoms.py",
                    str(input_file_mae),
                    ligand_file_raw_mae.name,
                    "-asl",
                    f"not res. {ligand_resname}" if ligand_resname else "not ligand",
                ]
            )
            subprocess.run(  # then only first molecule from potential ligands
                [
                    str(schrodinger_directory / "run"),
                    "delete_atoms.py",
                    ligand_file_raw_mae.name,
                    ligand_file_mae.name,
                    "-asl",
                    "mol. >1",
                ]
            )

        logger.debug("Merging protein and ligand in the right order ...")
        subprocess.run(
            [
                str(schrodinger_directory / "utilities/structcat"),
                "-i",
                protein_file_mae.name,
                ligand_file_mae.name,
                "-o",
                protein_ligand_file_mae.name,
            ]
        )

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
        grid_input_file.write(f"RECEP_FILE '{protein_ligand_file_mae.name}'\n")
        grid_input_file.write("LIGAND_INDEX 2\n")
        grid_input_file.flush()

        grid_file_path = Path(cache_dir) / (
            sha256_objects([input_file_mae, ligand_resname]) + ".zip"
        )  # caching via hash based on input structure and chosen ligand
        if grid_file_path.is_file():
            logger.debug("Found cached grid file ..")
        else:
            logger.debug("Generating grid for docking ...")
            subprocess.run(
                [
                    str(schrodinger_directory / "glide"),
                    grid_input_file.name,
                    "-HOST",
                    "localhost",
                    "-WAIT",
                    "-OVERWRITE",
                ]
            )
            shutil.move(
                str(Path(".") / (Path(grid_input_file.name).stem + ".zip")), grid_file_path
            )

        if logger.getEffectiveLevel() != 10:  # remove grid logs etc.
            paths = Path(".").glob(f"*{Path(grid_input_file.name).stem}*")
            for path in paths:
                path.unlink()

        logger.debug("Writing input file for docking ...")
        docking_input_file.write(f"GRIDFILE '{str(grid_file_path)}'\n")
        docking_input_file.write(f"LIGANDFILE '{mols_file_sdf.name}'\n")
        docking_input_file.write(f"LIGPREP True\n")
        docking_input_file.write("POSE_OUTTYPE ligandlib_sd\n")
        docking_input_file.write(f"COMPRESS_POSES False\n")
        docking_input_file.write(f"POSES_PER_LIG {n_poses}\n")
        docking_input_file.write(f"PRECISION {precision}\n")
        if shape_restrain:
            docking_input_file.write(f"SHAPE_RESTRAIN True\n")
            docking_input_file.write(f"SHAPE_REF_LIGAND_FILE '{ligand_file_mae.name}'\n")
        if macrocyles:
            docking_input_file.write(f"MACROCYCLE True\n")
        docking_input_file.flush()

        logger.debug("Running docking ...")
        subprocess.run(
            [
                str(schrodinger_directory / "glide"),
                docking_input_file.name,
                "-HOST",
                "localhost",
                "-WAIT",
                "-OVERWRITE",
            ]
        )

        logger.debug("Filtering poses for appropriate number ...")
        docking_input_file_path = Path(docking_input_file.name)
        sd_file_path = Path(".") / (docking_input_file_path.stem + "_lib.sdf")
        if not sd_file_path.is_file():
            logger.debug("No docking poses were generated during docking ...")
            return
        supplier = Chem.SDMolSupplier(str(sd_file_path), removeHs=False)
        sd_writer = Chem.SDWriter(str(output_file_sdf))
        mol_counter_dict = {}
        for mol in supplier:
            # SDF from glide is sorted by docking score, but mols are in mixed order
            name = mol.GetProp("_Name")
            if name not in mol_counter_dict.keys():
                mol_counter_dict[name] = 0
            if mol_counter_dict[name] < n_poses:
                sd_writer.write(mol)
                mol_counter_dict[name] += 1
        sd_file_path.unlink()  # manually delete file

        if logger.getEffectiveLevel() != 10:  # remove docking logs etc.
            paths = Path(".").glob(f"*{docking_input_file_path.stem}*")
            for path in paths:
                path.unlink()

    return
