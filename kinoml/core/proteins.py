"""
``MolecularComponent`` objects that represent protein-like entities.
"""
import logging

from .components import BaseProtein
from .sequences import Biosequence
from ..utils import APPDIR

logger = logging.getLogger(__name__)


class AminoAcidSequence(BaseProtein, Biosequence):
    """Biosequence that only allows proteinic aminoacids

    Parameters
    ----------
    sequence : str
        The FASTA sequence for this protein (one-letter symbols)
    name : str, optional
        Free-text identifier for the sequence
    """

    ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
    _ACCESSION_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=protein&id={}&rettype=fasta&retmode=text"

    def __init__(self, sequence, name="", *args, **kwargs):
        BaseProtein.__init__(self, name=name, *args, **kwargs)
        Biosequence.__init__(self)
