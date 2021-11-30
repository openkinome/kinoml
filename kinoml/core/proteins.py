"""
``MolecularComponent`` objects that represent protein-like entities.
"""
import logging

from .components import BaseProtein, BaseStructure
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


class UniprotProtein(BaseProtein):
    """
    A protein represented by its UniProt ID, uniquely.

    Parameters
    ----------
    uniprot_id : str
        Uniprot ID for this protein
    """

    def __init__(self, uniprot_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uniprot_id = uniprot_id


class ProteinStructure(BaseProtein, BaseStructure):
    """
    Structural representation of a protein

    Note
    ---
    This is probably going to be redone, so do not invest too much
    """

    @classmethod
    def from_file(cls, path, ext=None, **kwargs):

        from MDAnalysis import Universe
        from pathlib import Path

        identifier = Path(path).stem  #  set id to be the file name

        u = Universe(path)
        p = Path(path)

        return cls(
            pname=p.name,
            metadata={"path": path, "id": identifier},
            universe=u,
            **kwargs,
        )

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
        import requests
        import tempfile
        import MDAnalysis as mda
        from pathlib import Path
        from appdirs import user_cache_dir

        cached_path = Path(APPDIR.user_cache_dir)

        path = f"{cached_path}/{identifier}.pdb"

        url = f"https://files.rcsb.org/download/{identifier}.pdb"
        response = requests.get(url)

        with open(path, "wb") as pdb_file:  # saving the pdb to cache
            pdb_file.write(response.content)

        u = mda.Universe(path)

        return cls(metadata={"path": path, "id": identifier}, universe=u, **kwargs)

    @property
    def sequence(self):
        from MDAnalysis.lib.util import convert_aa_code

        pdb_seq = []
        three_l_codes = [convert_aa_code(i) for i in list(AminoAcidSequence.ALPHABET)]

        for r in self.universe.residues:
            if r.resname in three_l_codes:
                pdb_seq.append(convert_aa_code(r.resname))
            else:
                continue

        s = "".join(pdb_seq)

        return AminoAcidSequence(s)


class Kinase(ProteinStructure):

    """
    Extends ``Protein`` to provide kinase-specific methods of
    instantiation.

    Note
    ----
    TODO: Define role vs those objects under ``kinoml.core.kinase``
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
