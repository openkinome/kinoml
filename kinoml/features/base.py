
class _BaseFeaturizer:

    """
    Featurization API. Subclasses must implement ``self._featurize``, considering
    that ``self.molecule`` can represent slightly different object models:

        - ``kinoml.core.ligand.Ligand`` or ``kinoml.core.ligand.RDKitLigand``
        - ``kinoml.core.protein.Protein``
        - ``kinoml.core.complex.Complex``

    Parameters
    ==========
    molecule
        Depending on the implementation, small compound, protein, etc.
    """

    def __init__(self, molecule, *args, **kwargs):
        self.molecule = molecule

    def _featurize(self):
        raise NotImplementedError("Implement this method in your subclass")

    def featurize(self):
        """
        Main entry point for Featurization API. Check ``self._featurize``
        for implementation details.

        Returns
        =======
        featurized
            Featurized representation of ``self.molecule`` (implementation
            dependent).
        """
        return self._featurize()
