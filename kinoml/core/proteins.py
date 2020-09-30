import logging
from typing import List, Union

from .components import BaseProtein, BaseStructure
from .sequences import Biosequence
from ..utils import download_file, APPDIR

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
        if path.startswith("http"):
            from appdirs import user_cache_dir

            # TODO: where to save, how to name
            self.path = f"{user_cache_dir()}/{self.name}.{path.split('.')[-1]}"
            download_file(path, self.path)
        else:
            self.path = path
        if electron_density_path is not None:
            if electron_density_path.starswith("http"):
                from appdirs import user_cache_dir

                # TODO: where to save, how to name
                self.electron_density_path = (
                    f"{user_cache_dir()}/{self.name}.{path.split('.')[-1]}"
                )
                download_file(path, self.path)
            else:
                self.electron_density_path = electron_density_path


class PDBProtein(FileProtein):
    def __init__(self, pdb_id, path="", metadata=None, name="", *args, **kwargs):
        super().__init__(path=path, metadata=metadata, name=name, *args, **kwargs)
        from appdirs import user_cache_dir

        self.pdb_id = pdb_id
        self.path = (
            f"{user_cache_dir()}/{self.name}.pdb"  # TODO: if not available go for mmcif
        )
        download_file(f"https://files.rcsb.org/download/{pdb_id}.pdb", self.path)
        self.electron_density_path = f"{user_cache_dir()}/{self.name}.mtz"
        download_file(
            f"https://edmaps.rcsb.org/coefficients/{pdb_id}.mtz",
            self.electron_density_path,
        )

    @staticmethod
    def klifs_pocket(
        pdb_id: str, chain: Union[str or None] = None, altloc: Union[str or None] = None
    ) -> List[int]:
        """
        Read electron density from a file.
        Parameters
        ----------
        pdb_id: str
            PDB identifier.
        chain: str or None
            Chain identifier for PDB structure.
        altloc: str or None
            Alternate location identifier for PDB structure
        Returns
        -------
        pocket_resids: list of int
            A list of residue identifiers describing the pocket.
        """
        import klifs_utils

        structures_df = klifs_utils.remote.structures.structures_from_pdb_ids(
            pdb_id, chain=chain, alt=altloc
        )
        structure_id = structures_df.iloc[0]["structure_ID"]

        pocket_df = klifs_utils.remote.coordinates.pocket.mol2_to_dataframe(
            structure_id
        )
        pocket_resids = (
            pocket_df["subst_name"].str.slice(start=3).astype(int).unique().tolist()
        )

        return pocket_resids


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
