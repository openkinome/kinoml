class BaseMeasurement:
    """
    We will have several subclasses depending on the experiment
    They will also provide loss functions tailored to it.
    """

    pass


class Measured:
    def __init__(self, measurement=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.measurement = measurement

