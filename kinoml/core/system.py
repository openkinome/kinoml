from .protein import Protein
from .ligand import Ligand
from .measurements import Measured


class MolecularSystem:

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

    def is_complex(self):
        return len(self.ligand) == 1 and len(self.protein) == 1


class MeasuredMolecularSystem(MolecularSystem, Measured):

    """
    
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert hasattr(self, "components")
        assert hasattr(self, "measure")
