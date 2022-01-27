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
