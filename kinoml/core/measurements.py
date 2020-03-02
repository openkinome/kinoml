class BaseMeasurement:
    """
    We will have several subclasses depending on the experiment
    They will also provide loss functions tailored to it.

    Parameters
    ----------
    value : float, int, or any other numeric representation
        Not necessarily scalar! Some experiments can have more than
        one replicate.
    conditions : kinoml.core.conditions.AssayConditions-like
    components : list of kinoml.core.{protein,ligand,complex} objects
        The molecular entities measured

    TODO: Investigate possible uses for `pint`
    """

    def __init__(self, value, conditions, components, **kwargs):
        self.value = value
        self.conditions = conditions
        self.components = components


class PercentageDisplacementMeasurement(BaseMeasurement):
    def __init__(self, value, **kwargs):
        assert 0 <= value <= 100, f"Value {value} not in [0, 100]"
        super().__init__(self, value, **kwargs)

    def to_IC50(self):
        """
        Ideally, self.conditions should contain all we need to do
        the math here?
        """
        pass


class Measured:
    def __init__(self, measurement=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.measurement = measurement

