"""
Base classes for all MolecularComponents.

One or more components can form a System.
Proteins, ligands, and other molecular entities are
derived the base class ``MolecularComponent``.
"""


class MolecularComponent:
    """
    Abstract base molecular entity.
    """

    def __init__(self, name="", metadata=None, *args, **kwargs):
        self.name = name
        if metadata is None:
            metadata = {}
        self.metadata = metadata

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name}>"


class BaseStructure(MolecularComponent):
    """
    Draft class meant to contain 3D structures
    or trajectories of (macro)molecules. Wraps MDAnalysis'
    ``Universe`` objects.

    Parameters
    ----------
    universe : MDAnalysis.Universe
    """

    def __init__(self, universe, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.universe = universe


class BaseLigand(MolecularComponent):
    """
    Base class for all ligand-like entities.
    """


class BaseProtein(MolecularComponent):
    """
    Base class for all protein-like entities.
    """
