"""
``MolecularComponent`` objects that represent ligand-like entities.
"""

import logging

from openff.toolkit.topology import Molecule

from .components import BaseLigand


logger = logging.getLogger(__name__)


class Ligand(BaseLigand):

    def __init__(self, molecule=None, name=None, smiles=None, metadata=None, **kwargs):
        BaseLigand.__init__(self, name=name, metadata=metadata, **kwargs)
        self._molecule = molecule
        self._smiles = smiles

    @property
    def molecule(self):
        return self._molecule

    @molecule.setter
    def molecule(self, new_value):
        self._molecule = new_value

    @molecule.getter
    def molecule(self):
        if self._smiles:
            self._molecule = Molecule.from_smiles(
                smiles=self._smiles, allow_undefined_stereo=True
            )
            if self.name is None:
                self.name = self._smiles
            if self.metadata is None:
                self.metadata = {"smiles": self._smiles}
            else:
                self.metadata.update({"smiles": self._smiles})
        return self._molecule

    @classmethod
    def from_smiles(cls, smiles, name=None, allow_undefined_stereo=True, **kwargs):
        molecule = Molecule.from_smiles(
            smiles=smiles, allow_undefined_stereo=allow_undefined_stereo, **kwargs
        )
        if name is None:
            name = smiles
        return cls(smiles=smiles, molecule=molecule, name=name, metadata={"smiles": smiles})

    @classmethod
    def from_file(cls, file_path, name=None, allow_undefined_stereo=True, **kwargs):
        molecule = Molecule.from_file(
            file_path=file_path, allow_undefined_stereo=allow_undefined_stereo, **kwargs
        )
        if name is None:
            name = molecule.to_smiles(explicit_hydrogens=False)
        return cls(molecule=molecule, name=name, metadata={"file_path": file_path})
