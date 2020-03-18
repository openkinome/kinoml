from .protein import Protein
from .ligand import Ligand
from .measurements import Measured


class Complex:

    """
    Complex objects host one protein and one ligand, at least.
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

    def dock(self):
        raise NotImplementedError
