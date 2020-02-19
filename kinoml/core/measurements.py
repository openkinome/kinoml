class Measurement:
    pass


class Measured:
    def __init__(self, measure=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.measure = measure

