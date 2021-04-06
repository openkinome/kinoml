class MolecularComponent:
    """
    Abstract base molecular entity. Several components
    can form a System. Proteins, ligands, and other
    molecular entities are derived from this class.
    """

    def __init__(self, name="", metadata=None, *args, **kwargs):
        self.name = name
        if metadata is None:
            metadata = {}
        self.metadata = metadata

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name}>"


class BaseStructure(MolecularComponent):
    def __init__(self, universe, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.universe = universe


class BaseLigand(MolecularComponent):
    pass


class BaseProtein(MolecularComponent):
    pass
