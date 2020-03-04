import logging

from openforcefield.topology import Molecule

logger = logging.getLogger(__name__)


class Ligand(Molecule):

    """
    Small molecule object based on the OpenForceField toolkit.

    !!! todo
        Everything in this class
    """

    def __init__(self, _provenance=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if _provenance is None:
            _provenance = {}
        self._provenance = _provenance

    del __init__.__doc__

    @classmethod
    def from_smiles(cls, smiles, **kwargs):
        """
        Same as `openforcefield.topology.Molecule`, but adding
        information about the original SMILES to `._provenance` dict.
        """
        ligand = super().from_smiles(smiles, **kwargs)
        return cls(other=ligand, _provenance={"smiles": smiles})
