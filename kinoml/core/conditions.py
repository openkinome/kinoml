from typing import Union
import json


class BaseConditions:

    """
    Contains information about the experimental conditions.
    We ensure objects are immutable by using read-only properties
    for all attributes. Do NOT modify private attributes or
    hashing will break.

    Parameters:
        strict: Whether to perform sanity checks at initialization.

    """

    def __init__(self, strict: bool = True):
        if strict:
            self.sanity_checks()

    def sanity_checks(self):
        """
        Perform some checks for valid values
        """

    def _properties(self):
        """
        Return a dictionary with the classname and all defined properties.
        Used for equality comparisons in subclasses.
        """
        props = {"classname": self.__class__.__name__}
        for name in dir(self):
            if name.startswith("_"):
                continue
            clsattr = getattr(self.__class__, name)
            if isinstance(clsattr, property):
                props[name] = getattr(self, name)
        return props

    def __hash__(self):
        return hash(json.dumps(self._properties()))

    def __eq__(self, other):
        return self._properties() == other._properties()


class AssayConditions(BaseConditions):
    """
    Contains information about the experimental conditions
    of a given assay.

    Parameters:
        pH: Acidity conditions
    """

    def __init__(self, pH: Union[int, float] = 7.0, *args, **kwargs):
        self._pH = pH

        # Finish initialization
        super().__init__(*args, **kwargs)

    @property
    def pH(self):
        return self._pH

    def sanity_checks(self):
        super().sanity_checks()
        assert 0 <= self.pH <= 14, f"pH must be within [0, 14], but {self.pH} was specified"
