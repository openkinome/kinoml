from typing import Union


class BaseConditions:

    """
    Contains information about the experimental conditions.

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
        raise NotImplementedError


class AssayConditions(BaseConditions):
    """
    Contains information about the experimental conditions
    of a given assay.

    Parameters:
        pH: Acidity conditions

    """

    def __init__(self, pH: Union[int, float] = None, *args, **kwargs):
        self.pH = pH

        # Finish initialization
        super().__init__(*args, **kwargs)

    def sanity_checks(self):
        """
        Perform some checks for valid values
        """
        assert 0 <= self.pH <= 14, f"pH must be within [0, 14], but you supplied {self.pH}"
