class BaseConditions:
    pass


class AssayConditions(BaseConditions):
    """
    Contains information about the experimental conditions
    of a given assay.
    """

    def __init__(self, pH=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pH = pH
