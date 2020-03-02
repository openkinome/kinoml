from .protein import Protein
from .ligand import Ligand
from .measurements import Measured


class Complex:

    """

    """

    def __init__(self, components, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.components = components

    @property
    def ligand(self):
        for component in components:
            if isinstance(component, Ligand):
                yield component

    @property
    def protein(self):
        for component in components:
            if isinstance(component, Protein):
                yield component

    def sanity_check(self):  # this is a requirement
        return len(self.ligand) >= 1 and len(self.protein) >= 1

    def dock():
        pass


class MeasuredComplex(Complex, Measured):

    """

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert hasattr(self, "components")
        assert hasattr(self, "measure")

