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
        BaseLigand.__init__(self, name=name, _provenance=_provenance)
        Molecule.__init__(self, *args, **kwargs)

    del __init__.__doc__

    @classmethod
    def from_smiles(cls, smiles, name="", **kwargs):  # pylint: disable=arguments-differ
        """
        Same as `openforcefield.topology.Molecule`, but adding
        information about the original SMILES to `._provenance` dict.
        """
        ligand = super().from_smiles(smiles, **kwargs)
        if not name:
            name = smiles
        return cls(other=ligand, name=name, _provenance={"smiles": smiles})
