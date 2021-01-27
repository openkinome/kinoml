from __future__ import annotations

from typing import Iterable

from .components import MolecularComponent
from .ligands import BaseLigand
from .proteins import BaseProtein


class System:

    """
    System objects host one or more MolecularComponent objects,
    and, optionally, a measurement.

    Parameters:
        components: Molecular entities defining this system
        measurement: Optional measurement for this system.
        strict: Whether to perform sanity checks (default) or not.
    """

    def __init__(
        self, components: Iterable[MolecularComponent], strict: bool = True, *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.components = components
        self.featurizations = {}
        if strict:
            self.check()

    def _components_by_type(self, type_):
        for component in self.components:
            if isinstance(component, type_):
                yield component

    def check(self):
        assert self.components, "`System` must specify at least one component"

    @property
    def name(self) -> str:
        """
        Generates a readable name out of the components names
        """
        return " & ".join([c.name for c in self.components])

    @property
    def weight(self) -> float:
        """
        Calculate the molecular weight of the system

        Note: This is just an example on how/why this level of
        abstraction can be useful.
        """
        mass = 0
        for component in self.components:
            if not hasattr(component, "mass"):  # It will be unimplemented for some types!
                raise TypeError("This system contains at least one component without mass.")
            mass += component.mass
        return mass

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} with "
            f"{len(self.components)} components ({', '.join([repr(c) for c in self.components])})>"
        )


class Protein(System):
    """
    A system with at least one protein
    """

    @property
    def protein(self):
        return list(self._components_by_type(BaseProtein))[0]

    @property
    def proteins(self):
        return list(self._components_by_type(BaseProtein))

    def check(self):  # this is a requirement
        super().check()
        assert len(list(self.proteins)) >= 1, (
            "A Protein must specify at least one protein. "
            f"Current contents: {self}."
        )


class Ligand(System):
    """
    A system with at least one ligand
    """

    @property
    def ligand(self):
        return list(self._components_by_type(BaseLigand))[0]

    @property
    def ligands(self):
        return list(self._components_by_type(BaseLigand))

    def check(self):  # this is a requirement
        super().check()
        assert len(list(self.ligands)) >= 1, (
            "A Ligand must specify at least one ligand. "
            f"Current contents: {self}."
        )


class ProteinLigandComplex(Protein, Ligand):
    """
    A system with at least one protein and one ligand
    """

    def check(self):  # this is a requirement
        super().check()
        assert len(list(self.ligands)) >= 1 and len(list(self.proteins)) >= 1, (
            "A ProteinLigandComplex must specify at least one ligand and one protein. "
            f"Current contents: {self}."
        )
