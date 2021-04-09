"""
Each Measurement object can store a ``conditions``
attribute which should contain one of the classes
here defined.

For example, experimental measurements can have an
``AssayConditions`` object specifying the variables
involved in the experiment, like pH.
"""

from typing import Union
import json


class BaseConditions:

    """
    Contains information about the experimental conditions.
    We ensure objects are immutable by using read-only properties
    for all attributes. Do NOT modify private attributes or
    hashing will break.

    Parameters
    ----------
    strict : bool, optional=True
        Whether to perform safety checks at initialization.
    """

    def __init__(self, strict: bool = True):
        if strict:
            self.check()

    def check(self):
        """
        Perform some checks for valid values
        """

    def _properties(self, classname: bool = True) -> dict:
        """
        Return a dictionary with the classname and all defined properties.
        Used for equality comparisons in subclasses.

        Parameters
        ----------
        classname : bool, optional=True
            Whether to include the name of the instance class

        Returns
        -------
        dict
        """
        props = {"classname": self.__class__.__name__} if classname else {}
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

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"{' '.join([f'{k}={v}' for k, v in self._properties(classname=False).items()])}>"
        )


class AssayConditions(BaseConditions):
    """
    Contains information about the experimental conditions
    of a given assay.

    Parameters
    ----------
    pH : int or float, optional=7.0
        Acidity conditions
    """

    def __init__(self, pH: Union[int, float] = 7.0, *args, **kwargs):
        self._pH = pH

        # Finish initialization
        super().__init__(*args, **kwargs)

    @property
    def pH(self):
        return self._pH

    def check(self):
        super().check()
        assert 0 <= self.pH <= 14, f"pH must be within [0, 14], but {self.pH} was specified"
