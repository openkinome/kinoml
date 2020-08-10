import logging

from openforcefield.topology import Molecule

from .components import BaseLigand

logger = logging.getLogger(__name__)


class SmilesLigand(BaseLigand):
    def __init__(self, smiles, metadata=None, name="", *args, **kwargs):
        BaseLigand.__init__(self, name=name, metadata=metadata)
        self.smiles = smiles


class FileLigand(BaseLigand):
    def __init__(self, path, metadata=None, name="", *args, **kwargs):
        BaseLigand.__init__(self, name=name, metadata=metadata)
        if path.startswith("http"):
            print("We would download this and save it to appdir.user_cache")
        self.path = path


class Ligand(BaseLigand, Molecule):

    """
    Small molecule object based on the OpenForceField toolkit.

    !!! todo
        Everything in this class
    """

    def __init__(self, metadata=None, name="", *args, **kwargs):
        Molecule.__init__(self, *args, **kwargs)
        BaseLigand.__init__(self, name=name, metadata=metadata)

    @classmethod
    def from_smiles(
        cls, smiles, name=None, allow_undefined_stereo=True, **kwargs
    ):  # pylint: disable=arguments-differ
        """
        Same as `openforcefield.topology.Molecule`, but adding
        information about the original SMILES to `.metadata` dict.

        !!! todo
            Inheritance from these methods in OFF is broken because they
            delegate directly to the underlying toolkits, and type is
            lost on the way, so we will always obtain an
            openforcefield.topology.Molecule, no matter what.

            PR #583 has been submitted to patch upstream
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
