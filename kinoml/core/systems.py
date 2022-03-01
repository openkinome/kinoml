"""
``System`` objects define a collection of related
``MolecularComponent`` instances. They are normally
attached to a ``Measurement``, and, in the context
of a machine learning exercise, will be featurized
with different classes found under ``kinoml.features``.
Featurization turns a ``System`` into a tensor-like
object, like Numpy arrays.
"""
from __future__ import annotations

from typing import Iterable

from .components import MolecularComponent
from .ligands import BaseLigand
from .proteins import BaseProtein


class System:

    """
    System objects host one or more MolecularComponent.

    Parameters
    ----------
    components : list of MolecularComponent
        Molecular entities defining this system
    strict: bool, optional=True
        Whether to perform sanity checks (default) or not.

    Attributes
    ----------
    featurizations : dict
        This dictionary will store the different featurization
        steps a ``System`` is submitted to. The keys for this
        dictionary are usually the *name* of the featurizer
        class. Additionally, a ``Pipeline`` might define
        a ``last`` key, indicating that particular object
        was the final result of a chain of featurizers.
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
        """
        Yield MolecularComponent objects of a given type only
        """
        for component in self.components:
            if isinstance(component, type_):
                yield component

    def check(self):
        assert self.components, "`System` must specify at least one component"
        return True

    @property
    def name(self) -> str:
        """
        Generates a readable name out of the components names
        """
        return " & ".join([str(c.name) for c in self.components])

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


class ProteinSystem(System):
    """
    A System that contains Protein objects. It defines two properties:

    - ``protein``: get the first Protein found in the components
    - ``proteins``: get all Protein objects found in the components
    """

    @property
    def protein(self):
        return next(self._components_by_type(BaseProtein))

    @property
    def proteins(self):
        return list(self._components_by_type(BaseProtein))

    def check(self):  # this is a requirement
        super().check()
        assert (
            len(self.proteins) >= 1
        ), f"A ProteinSystem must specify at least one Protein. Current contents: {self}."
        return True


class LigandSystem(System):
    """
    A System that contains Ligand objects. It defines two properties:

    - ``ligand``: get the first Ligand found in the components
    - ``ligands``: get all Ligand objects found in the components
    """

    @property
    def ligand(self):
        return next(self._components_by_type(BaseLigand))

    @property
    def ligands(self):
        return list(self._components_by_type(BaseLigand))

    def check(self):  # this is a requirement
        super().check()
        assert (
            len(self.ligands) >= 1
        ), f"A LigandSystem must specify at least one Ligand. Current contents: {self}."
        return True


class ProteinLigandComplex(ProteinSystem, LigandSystem):
    """
    A system with at least one protein and one ligand
    """

    def check(self):
        assert ProteinSystem.check(self) and LigandSystem.check(self), (
            "A ProteinLigandComplex must specify at least one Protein and one Ligand. "
            f"Current contents: {self}"
        )
