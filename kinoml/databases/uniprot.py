from pathlib import Path
from typing import Union

from appdirs import user_cache_dir


def download_fasta_file(
    uniprot_id: str, directory: Union[Path, str] = user_cache_dir()
) -> Union[Path, False]:
    """
    Download a fasta file for a given UniProt identifier.

    Parameters
    ----------
    uniprot_id: str
        The UniProt entry of interest.
    directory: Path or str
        The path to a directory for saving the file.

    Returns
    -------
    : Path or False
        The path to the downloaded file, False if not successful.
    """
    from ..utils import download_file

    fasta_path = Path(directory) / f"{uniprot_id}.fasta"
    if fasta_path.is_file():
        return fasta_path

    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    if download_file(url, fasta_path):
        return fasta_path

    return False
