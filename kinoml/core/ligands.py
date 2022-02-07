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
            self._smiles = None  # remove smiles
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


class Ligand2(BaseLigand, Molecule):

    def __init__(self, molecule=None, smiles=None, name=None, metadata=None, *args, **kwargs):
        Molecule.__init__(self, molecule, *args, **kwargs)
        BaseLigand.__init__(self, name=name, metadata=metadata)
        self._smiles = smiles

    @classmethod
    def from_smiles(cls, smiles, name=None, allow_undefined_stereo=True, **kwargs):
        """
        Same as `openff.toolkit.topology.Molecule`, but adding information about the original
        SMILES to the ``.metadata`` dict.

        Parameters
        ----------
        smiles: str
            SMILES representation of the ligand. This string will be stored in the ``metadata``
            attribute under the ``smiles`` key.
        name: str, optional
            An easily identifiable name for the molecule. If not given, ``smiles`` is used.
        """
        self = super().from_smiles(smiles, allow_undefined_stereo=allow_undefined_stereo, **kwargs)
        if name is None:
            name = smiles
        super().__init__(self, name=name, metadata={"smiles": smiles})
        return self

    @classmethod
    def from_file(cls, file_path, name=None, allow_undefined_stereo=True, **kwargs):
        """
        Same as `openff.toolkit.topology.Molecule`, but adding information about the file to
        the ``.metadata`` dict.

        Parameters
        ----------
        file_path: str or Path
            Path to file of the ligand. This string will be stored in the ``metadata`` attribute
            under the ``file_path`` key.
        name: str, optional
            An easily identifiable name for the molecule. If not given, ``file_path`` is used.
        """
        self = super().from_file(
            file_path, allow_undefined_stereo=allow_undefined_stereo, **kwargs
        )
        if name is None:
            name = file_path
        super().__init__(self, name=name, metadata={"file_path": file_path})
        return self

    def to_dict(self):
        """Dict representation of the Molecule, including the ``metadata`` dictionary."""
        d = super().to_dict()
        d["metadata"] = self.metadata.copy()
        return d

    def _initialize_from_dict(self, molecule_dict):
        """Same as Molecule's method, but including the ``metadata`` dict."""
        super()._initialize_from_dict(molecule_dict)
        self.metadata = molecule_dict["metadata"].copy()
