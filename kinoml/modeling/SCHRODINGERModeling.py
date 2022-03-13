import logging
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile
from typing import Union


logger = logging.getLogger(__name__)


def run_prepwizard(
    schrodinger_directory: Union[Path, str],
    input_file: Union[Path, str],
    output_file: Union[Path, str],
    cap_termini: bool = True,
    build_loops: bool = True,
    sequence: Union[str, None] = None,
    chain_id: str = "",
    protein_pH: str = "neutral",
    propka_pH: float = 7.4,
    epik_pH: float = 7.4,
    force_field: str = "3",
):
    """
    Run the prepwizard utility to prepare a protein structure.

    Parameters
    ----------
    schrodinger_directory: Path or str
        The path to the directory of the Schrodinger installation.
    input_file: Path or str
        The path to the input file.
    output_file: Path or str
        The path to the output file.
    cap_termini: bool, default=True
        If termini should be capped.
    build_loops: bool, default=True
        If loops should be built.
    sequence: str or None
        The amino acid sequence in single letter codes that should be used for loop building.
        Also needs the chain_id parameter to work correctly.
    chain_id: str, default=""
        The chain ID of the protein that should be modeled based on the given sequence.
    protein_pH: str, default='neutral'
        The pH used during protonation of the protein ('very_low', 'low', 'neutral', 'high').
    propka_pH: float, default=7.4
        Run PROPKA at given pH.
    epik_pH: float, default=7.4
        The pH used during protonation of the ligand.
    force_field: str, default='3'
        Force field to use during minimization (2005, 3)
    """
    schrodinger_directory = Path(schrodinger_directory)
    executable = str(schrodinger_directory / "utilities/prepwizard")
    standard_arguments = [
        str(input_file), str(output_file), "-HOST", "localhost", "-WAIT", "-keepfarwat",
        "-disulfides", "-glycosylation", "-palmitoylation", "-mse", "-fillsidechains",
        "-samplewater", "-pH", protein_pH, "-propka_pH", str(propka_pH), "-minimize_adj_h",
        "-epik_pH", str(epik_pH), "-f", force_field,
    ]
    optional_arguments = []
    if cap_termini:
        optional_arguments.append("-c")
    if build_loops:
        optional_arguments.append("-fillloops")

    if sequence:  # one letter characters, 60 per line, no header
        with NamedTemporaryFile(mode="w", suffix=".fasta") as fasta_file:
            sequence = "\n".join([sequence[i:i + 60] for i in range(0, len(sequence), 60)])
            fasta_file.write(f">entry:{chain_id}\n")
            fasta_file.write(sequence)
            fasta_file.flush()
            subprocess.run(
                [executable] + standard_arguments + optional_arguments + ["-fasta_file", fasta_file.name]
            )
    else:
        subprocess.run(
            [executable] + standard_arguments + optional_arguments
        )

    if logger.getEffectiveLevel() != logging.DEBUG:  # remove prepwizard log
        paths = Path(".").glob(f"*{Path(input_file).stem}*")
        for path in paths:
            try:
                path.unlink()
            except FileNotFoundError:
                # may happen in multiprocessing of the same structure
                pass

    return


def mae_to_pdb(
        schrodinger_directory: Union[str, Path],
        mae_file_path: Union[str, Path],
        pdb_file_path: Union[str, Path]
):
    """
    Convert a structure file from MAE to PDB format.

    Parameters
    ----------
    schrodinger_directory: str or pathlib.Path
        The path to the directory of the Schrodinger installation.
    mae_file_path: str or pathlib.Path
        The path to the input file in MAE format.
    pdb_file_path: str or pathlib.Path
        The path to the output file in PDB format.
    """
    schrodinger_directory = Path(schrodinger_directory)
    arguments = [
        str(schrodinger_directory / "utilities/pdbconvert"),  # executable
        "-imae", str(mae_file_path), "-opdb", str(pdb_file_path),  # file paths
    ]
    subprocess.run(arguments)
    return


def shape_screen(
    schrodinger_directory: Union[Path, str],
    query_path: Union[str, Path],
    library_path: Union[str, Path],
    output_sdf_path: Union[str, Path],
    flexible: bool = True,
    thorough_sampling: bool = True,
    keep_best_match_only: bool = True,
):
    import gzip
    import shutil

    schrodinger_directory = Path(schrodinger_directory)
    executable = str(schrodinger_directory / "shape_screen")
    standard_arguments = [
        "-shape", str(query_path), "-screen", str(library_path), "-osd", "-atomtypes", "qsar",
        "-HOST", "localhost", "-WAIT",
    ]
    optional_arguments = []
    if flexible:
        optional_arguments.append("-flex")
        if thorough_sampling:
            optional_arguments += ["-sample", "thorough"]
    if keep_best_match_only:
        optional_arguments.append("-best")

    subprocess.run(
        [executable] + standard_arguments + optional_arguments
    )
    if logger.getEffectiveLevel() != logging.DEBUG:  # remove shape_screen log and okay
        paths = [
            Path(".") / f"{Path(query_path).stem}_shape.log",
            Path(".") / f"{Path(query_path).stem}_shape.okay"
        ]
        for path in paths:
            try:
                path.unlink()
            except FileNotFoundError:
                # may happen in multiprocessing of the same query file
                pass

    logger.debug("Unzipping and renaming results ...")
    output_sdfgz_path = Path(".") / f"{Path(query_path).stem}_align.sdfgz"
    with gzip.open(output_sdfgz_path, "rb") as sdfgz:
        with open(output_sdf_path, "wb") as sdf:
            shutil.copyfileobj(sdfgz, sdf)
    output_sdfgz_path.unlink()

    return
