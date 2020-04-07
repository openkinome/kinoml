from typing import Union, Iterable, Any

import numpy as np

from .conditions import AssayConditions
from .components import MolecularComponent


class BaseMeasurement:
    """
    We will have several subclasses depending on the experiment.
    They will also provide loss functions tailored to it.

    Values of the measurement can have more than one replicate. In fact,
    single replicates are considered a specific case of a multi-replicate.

    Parameters:
        values: The numeric measurement(s)
        conditions: Experimental conditions of this measurement
        components: Molecular entities measured
        strict: Whether to perform sanity checks at initialization.

    !!! todo
        Investigate possible uses for `pint`
    """

    def __init__(
        self,
        values: Union[float, Iterable[float]],
        conditions: AssayConditions,
        components: Iterable[MolecularComponent],
        strict: bool = True,
        **kwargs,
    ):
        self._values = np.asarray(values)
        self.conditions = conditions
        # TODO: Do we want `components` here? It might introduce cyclic dependencies.
        self.components = components

        if strict:
            self.sanity_checks()

    @property
    def values(self):
        return self._values

    def sanity_checks(self):
        """
        Perform some checks for valid values
        """
        pass

    def __eq__(self, other):
        return (
            (self.values == other.values).all()
            and self.conditions == other.conditions
            and self.components == other.components
        )


class PercentageDisplacementMeasurement(BaseMeasurement):

    """
    Measurement where the value(s) must be percentage(s) of displacement.
    """

    def sanity_checks(self):
        super().sanity_checks()
        assert (0 <= self.values <= 100).all(), f"One or more values are not in [0, 100]"

    def to_IC50(self):
        """
        Ideally, `self.conditions` should contain all we need to do
        the math here?
        """
        pass
