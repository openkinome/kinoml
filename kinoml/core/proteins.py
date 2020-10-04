import logging

from .components import BaseProtein, BaseStructure
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
    def __init__(
        self, path, electron_density_path=None, metadata=None, name="", *args, **kwargs
    ):
        super().__init__(name=name, metadata=metadata, *args, **kwargs)
        if str(path).startswith("http"):
            from appdirs import user_cache_dir

            # TODO: where to save, how to name
            self.path = f"{user_cache_dir()}/{self.name}.{path.split('.')[-1]}"
            download_file(path, self.path)
        else:
            self.path = path
        self.electron_density_path = electron_density_path
        if electron_density_path is not None:
            if electron_density_path.starswith("http"):
                from appdirs import user_cache_dir

                # TODO: where to save, how to name
                self.electron_density_path = (
                    f"{user_cache_dir()}/{self.name}.{path.split('.')[-1]}"
                )
                download_file(path, self.path)


class PDBProtein(FileProtein):

    def __init__(self, pdb_id, path="", metadata=None, name="", *args, **kwargs):
        super().__init__(path=path, metadata=metadata, name=name, *args, **kwargs)
        from ..utils import LocalFileStorage

        self.pdb_id = pdb_id
        self.path = LocalFileStorage.rcsb_structure_pdb(pdb_id)
        self.electron_density_path = LocalFileStorage.rcsb_electron_density_mtz(pdb_id)


class ProteinStructure(BaseProtein, BaseStructure):
    """
    Structural representation of a protein

    !!! todo
        This is probably going to be redone, so do not invest too much
    """

    @classmethod
    def from_file(cls, path, ext=None, **kwargs):
        from MDAnalysis import Universe
        from pathlib import Path

        u = Universe(path)
        p = Path(path)
        return cls(name=p.name, metadata={"path": path}, universe=u, **kwargs)

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
