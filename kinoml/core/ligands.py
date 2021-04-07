import logging
from openff.toolkit.topology import Molecule

from .components import BaseLigand
from ..utils import download_file

logger = logging.getLogger(__name__)


class SmilesLigand(BaseLigand):
    def __init__(self, smiles, metadata=None, name="", *args, **kwargs):
        BaseLigand.__init__(self, name=name, metadata=metadata)
        self.smiles = smiles


class FileLigand(BaseLigand):
    def __init__(self, path, metadata=None, name="", *args, **kwargs):
        super().__init__(name=name, metadata=metadata, *args, **kwargs)
        if str(path).startswith("http"):
            from appdirs import user_cache_dir

            # TODO: where to save, how to name
            self.path = f"{user_cache_dir()}/{self.name}.{path.split('.')[-1]}"
            download_file(path, self.path)
        else:
            self.path = path


class PDBLigand(FileLigand):
    def __init__(self, pdb_id, path, metadata=None, name="", *args, **kwargs):
        super().__init__(path, metadata=metadata, name=name, *args, **kwargs)
        from appdirs import user_cache_dir

        self.pdb_id = pdb_id
        self.path = f"{user_cache_dir()}/{self.name}.sdf"
        download_file(f"https://files.rcsb.org/ligands/view/{pdb_id}_ideal.sdf", self.path)


class OpenForceFieldLigand(BaseLigand, Molecule):

    """
    Small molecule object based on the OpenForceField toolkit.
    """

    def __init__(self, metadata=None, name="", *args, **kwargs):
        Molecule.__init__(self, *args, **kwargs)
        BaseLigand.__init__(self, name=name, metadata=metadata)

    @classmethod
    def from_smiles(
        cls, smiles, name=None, allow_undefined_stereo=True, **kwargs
    ):  # pylint: disable=arguments-differ
        """
        Same as `openff.toolkit.topology.Molecule`, but adding
        information about the original SMILES to ``.metadata`` dict.
        """
        self = super().from_smiles(smiles, allow_undefined_stereo=allow_undefined_stereo, **kwargs)
        if name is None:
            name = smiles
        super().__init__(self, name=name, metadata={"smiles": smiles})
        return self

    def to_dict(self):
        d = super().to_dict()
        d["metadata"] = self.metadata.copy()
        return d

    def _initialize_from_dict(self, molecule_dict):
        super()._initialize_from_dict(molecule_dict)
        self.metadata = molecule_dict["metadata"].copy()


# Alias OpenForceFieldLigand to Ligand
Ligand = OpenForceFieldLigand


class OpenForceFieldLikeLigand(BaseLigand):
    def __init__(self, molecule, metadata=None, name="", *args, **kwargs):
        super().__init__(name=name, metadata=metadata)
        self._molecule = molecule

    def __getattr__(self, attr):
        return getattr(self._molecule, attr)

    @classmethod
    def from_smiles(cls, smiles, name=None, **kwargs):
        raise NotImplementedError("Use ``OpenForceFieldLigand`` or implement API in a subclass")

    def to_rdkit(self):
        raise NotImplementedError("Use ``OpenForceFieldLigand`` or implement API in a subclass")

    def to_smiles(self):
        raise NotImplementedError("Use ``OpenForceFieldLigand`` or implement API in a subclass")


class RDKitLigand(OpenForceFieldLikeLigand):

    """
    Wrapper for RDKit molecules using some parts of the OpenForceField API

    .. warning::

        Implement other parts of the OFF Molecule API
    """

    @classmethod
    def from_smiles(cls, smiles, name=None, **kwargs):  # pylint: disable=arguments-differ
        """"""
        from rdkit.Chem import MolFromSmiles

        molecule = MolFromSmiles(smiles)
        if name is None:
            name = smiles
        return cls(molecule, name=name, metadata={"smiles": smiles})

    def to_rdkit(self):
        return self._molecule

    def to_smiles(self):
        """
        Return canonicalized SMILES

        More info: https://www.rdkit.org/docs/GettingStartedInPython.html#writing-molecules
        """
        from rdkit.Chem import MolToSmiles

        return MolToSmiles(self._molecule)