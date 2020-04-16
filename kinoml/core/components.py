class MolecularComponent:
    """
    Abstract base molecular entity. Several components
    can form a System. Proteins, ligands, and other
    molecular entities are derived from this class.
    """

    def __init__(self, name="", _provenance=None, *args, **kwargs):
        self.name = name
        if _provenance is None:
            _provenance = {}
        self._provenance = _provenance

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name}>"


class BaseLigand(MolecularComponent):
    pass


class BaseProtein(MolecularComponent):
    pass
