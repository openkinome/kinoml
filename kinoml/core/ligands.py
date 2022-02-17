"""
``MolecularComponent`` objects that represent ligand-like entities.
"""

import logging
from pathlib import Path
from typing import Union

from openff.toolkit.topology import Molecule

from .components import BaseLigand


logger = logging.getLogger(__name__)


class Ligand(BaseLigand):
    """
    Create a new Ligand object. An openff representation is accessible via the molecule attribute.

    Examples
    --------

    Create a ligand from file:

    >>> ligand = Ligand.from_file("data/molecules/chloroform.sdf", name="chloroform")

    Create a ligand from an openff molecule:

    >>> from openff.toolkit.topology import Molecule
    >>> molecule = Molecule.from_file("data/molecules/chloroform.sdf")
    >>> ligand = Ligand(molecule=molecule, name="chloroform")

    Create a ligand from SMILES:

    >>> ligand = Ligand.from_smiles("C(Cl)(Cl)Cl", name="chloroform")

    Create a ligand from SMILES with lazy instantiation:

    >>> ligand = Ligand(smiles="C(Cl)(Cl)Cl", name="chloroform")

    """
    def __init__(
            self,
            molecule: Union[Molecule, None] = None,
            smiles: str = "",
            name: str = "",
            metadata: Union[dict, None] = None,
            **kwargs
    ):
        """
        Create a new Ligand object. Lazy instantiation is possible via the smiles parameter.

        Parameters
        ----------
        molecule: openff.toolkit.topology.Molecule or None, default=None
            An openff representation of the ligand.
        smiles: str, default=""
            The SMILES representation of the ligand. Can be used for lazy instantiation, i.e. will
            interpreted when calling the molecule attribute the first time.
        name: str, default=""
            The name of the ligand.
        metadata: dict or None, default=None
            Additional metadata of the needed for e.g. featurizers or provenance.
        """
        BaseLigand.__init__(self, name=name, metadata=metadata, **kwargs)
        self._molecule = molecule
        self._smiles = smiles

    @property
    def molecule(self):
        """Decorate molecule to modify setter and getter."""
        return self._molecule

    @molecule.setter
    def molecule(self, new_value: Union[Molecule, None]):
        """
        Store a new value for molecule in the _molecule attribute.

        Parameters
        ----------
        new_value: openff.toolkit.topology.Molecule or None
            The new openff molecule.
        """
        self._molecule = new_value

    @molecule.getter
    def molecule(self):
        """
        Get the _molecule attribute. If the _smiles attribute is given and _molecule is None, a
        new openff molecule will be created from smiles, e.g. in case of lazy instantiation.
        """
        if not self._molecule and self._smiles:
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
    def from_smiles(
            cls,
            smiles: str,
            name: str = "",
            allow_undefined_stereo: bool = True,
            **kwargs
    ):
        """
        Create a Ligand from a SMILES representation.

        Parameters
        ----------
        smiles: str
            smiles: str
            The SMILES representation of the ligand.
        name: str, default=""
            The name of the ligand.
        allow_undefined_stereo: bool, default=True
            If undefined stereo centers should be allowed.
        **kwargs:
            Any keyword arguments allowed for the from_smiles method of the openff molecule class.
        """
        molecule = Molecule.from_smiles(
            smiles=smiles, allow_undefined_stereo=allow_undefined_stereo, **kwargs
        )
        if name is None:
            name = smiles
        return cls(molecule=molecule, name=name, metadata={"smiles": smiles})

    @classmethod
    def from_file(
            cls,
            file_path: Union[Path, str],
            name: str = "",
            allow_undefined_stereo: bool = True,
            **kwargs
    ):
        """
        Create a Ligand from file.

        Parameters
        ----------
        file_path: pathlib.Path or str
            The path to the molecular file. For supported formats see the openff molecule
            documentation.
        name: str, default=""
            The name of the ligand.
        allow_undefined_stereo: bool, default=True
            If undefined stereo centers should be allowed.
        **kwargs:
            Any keyword arguments allowed for the from_file method of the openff molecule class.
        """
        molecule = Molecule.from_file(
            file_path=file_path, allow_undefined_stereo=allow_undefined_stereo, **kwargs
        )
        if name is None:
            name = molecule.to_smiles(explicit_hydrogens=False)
        return cls(molecule=molecule, name=name, metadata={"file_path": file_path})
