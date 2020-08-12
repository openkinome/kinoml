import logging

from .components import BaseProtein
from .sequences import Biosequence
from ..utils import download_file

logger = logging.getLogger(__name__)


class AminoAcidSequence(BaseProtein, Biosequence):
    """Biosequence that only allows proteinic aminoacids"""

    ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
    _ACCESSION_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=protein&id={}&rettype=fasta&retmode=text"

    def __init__(self, sequence, name="", *args, **kwargs):
        BaseProtein.__init__(self, name=name, *args, **kwargs)
        Biosequence.__init__(self)


class FileProtein(BaseProtein):
    def __init__(self, path, metadata=None, name="", *args, **kwargs):
        BaseProtein.__init__(self, name=name, metadata=metadata)
        if path.startswith("http"):
            from appdirs import user_cache_dir

            # TODO: where to save, how to name
            self.path = f"{user_cache_dir()}/{self.name}.{path.split('.')[-1]}"
            download_file(path, self.path)
        else:
            self.path = path


class PDBProtein(FileProtein):
    def __init__(self, pdb_id, metadata=None, name="", *args, **kwargs):
        from appdirs import user_cache_dir

        FileProtein.__init__(self, path="", name=name, metadata=metadata)
        self.pdb_id = pdb_id
        self.path = f"{user_cache_dir()}/{self.name}.pdb"  # TODO: if not available go for mmcif
        download_file(f"https://files.rcsb.org/download/{pdb_id}.pdb", self.path)
        self.electron_density_path = f"{user_cache_dir()}/{self.name}.mtz"
        download_file(f"https://edmaps.rcsb.org/coefficients/{pdb_id}.mtz", self.electron_density_path)


class ProteinStructure(BaseProtein):
    """
    Structural representation of a protein

    !!! todo
        This is probably going to be redone, so do not invest too much
    """

    @classmethod
    def from_file(cls, path, ext=None, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_sequence(cls, sequence, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_uniprot(cls, identifier, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_hgnc(cls, identifier, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_name(cls, identifier, **kwargs):
        raise NotImplementedError

    @property
    def sequence(self):
        s = "".join([r.symbol for r in self.residues])
        return AminoAcidSequence(s)


class Kinase(ProteinStructure):

    """
    Extends `Protein` to provide kinase-specific methods of
    instantiation.
    """

    @classmethod
    def from_klifs(cls, identifier, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_kinmap(cls, identifier, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_manning(cls, identifier, **kwargs):
        raise NotImplementedError
