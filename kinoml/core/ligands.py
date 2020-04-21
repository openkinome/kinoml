import logging

from openforcefield.topology import Molecule

from .components import BaseLigand

logger = logging.getLogger(__name__)


class Ligand(BaseLigand, Molecule):

    """
    Small molecule object based on the OpenForceField toolkit.

    !!! todo
        Everything in this class
    """

    def __init__(self, _provenance=None, name="", *args, **kwargs):
        Molecule.__init__(self, *args, **kwargs)
        BaseLigand.__init__(self, name=name, _provenance=_provenance)

    @classmethod
    def from_smiles(cls, smiles, name=None, **kwargs):  # pylint: disable=arguments-differ
        """
        Same as `openforcefield.topology.Molecule`, but adding
        information about the original SMILES to `._provenance` dict.

        !!! todo
            Inheritance from these methods in OFF is broken because they
            delegate directly to the underlying toolkits, and type is
            lost on the way, so we will always obtain an
            openforcefield.topology.Molecule, no matter what.

            PR #583 has been submitted to patch upstream
        """
        self = super().from_smiles(smiles, **kwargs)
        if name is None:
            name = smiles
        super().__init__(self, name=name, _provenance={"smiles": smiles})
        return self

    def to_dict(self):
        d = super().to_dict()
        d["_provenance"] = self._provenance.copy()
        return d

    def _initialize_from_dict(self, molecule_dict):
        super()._initialize_from_dict(molecule_dict)
        self._provenance = molecule_dict["_provenance"].copy()
