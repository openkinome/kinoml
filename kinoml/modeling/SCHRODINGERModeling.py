from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile
from typing import Union


def run_prepwizard(
    schrodinger_directory: Union[Path, str],
    input_file: Union[Path, str],
    output_file: Union[Path, str],
    cap_termini: bool = True,
    build_loops: bool = True,
    sequence: Union[str, None] = None,
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
            fasta_file.write(sequence)
            fasta_file.flush()
            subprocess.run(
                [executable] + standard_arguments + optional_arguments + ["-fasta_file", fasta_file.name]
            )
    else:
        for arg in [executable] + standard_arguments + optional_arguments:
            print(arg, type(arg))
        subprocess.run(
            [executable] + standard_arguments + optional_arguments
        )

    return
