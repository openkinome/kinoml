import logging

from openforcefield.topology import Molecule

logger = logging.getLogger(__name__)


class Ligand(Molecule):

    """
    TODO: Everything in this class
    """

    def __init__(self, _provenance=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if _provenance is None:
            _provenance = {}
        self._provenance = _provenance

    @classmethod
    def from_smiles(cls, smiles, **kwargs):
        ligand = super().from_smiles(smiles, **kwargs)
        return cls(other=ligand, _provenance={"smiles": smiles})
