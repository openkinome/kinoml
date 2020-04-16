from typing import Union

from .components import MolecularComponent
from .ligands import BaseLigand
from .measurements import BaseMeasurement
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
        self,
        components: [MolecularComponent],
        measurement: Union[None, BaseMeasurement] = None,
        strict: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._measurement = None
        self.components = components
        self.measurement = measurement
        if strict:
            self.sanity_checks()

    @property
    def measurement(self):
        return self._measurement

    @measurement.setter
    def measurement(self, value):
        assert value is None or isinstance(value, BaseMeasurement)
        self._measurement = value

    def sanity_checks(self):
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
            mass += component.mass  # It will be unimplemented for some types!
        return mass

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} with "
            f"{len(self.components)} components ({', '.join([repr(c) for c in self.components])}) "
            f"and {self.measurement!r}>"
        )


class ProteinLigandComplex(System):
    """
    A system with at least one protein and one ligand
    """

    @property
    def ligand(self):
        for component in self.components:
            if isinstance(component, BaseLigand):
                return component

    @property
    def protein(self):
        for component in self.components:
            if isinstance(component, BaseProtein):
                return component

    @property
    def ligands(self):
        for component in self.components:
            if isinstance(component, BaseLigand):
                yield component

    @property
    def proteins(self):
        for component in self.components:
            if isinstance(component, BaseProtein):
                yield component

    def sanity_checks(self):  # this is a requirement
        super().sanity_checks()
        assert (
            len(list(self.ligands)) >= 1 and len(list(self.proteins)) >= 1
        ), "A ProteinLigandComplex must specify at least one Ligand and one Protein"

    # Bonus perks!
    def dock(self):
        raise NotImplementedError


class MeasuredSystem(System):

    """
    Subclass of System that requires a non-null `measurement` attribute.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.measurement is None:
            raise ValueError("`MeasuredSystem` must specify a non-null measurement.")
