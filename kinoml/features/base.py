
class _BaseFeaturizer:

    """
    Featurization API. Subclasses must implement ``self._featurize``, considering
    that ``self.molecule`` can represent slightly different object models:

        - ``kinoml.core.ligand.Ligand``
        - ``kinoml.core.protein.Protein``
        - ``kinoml.core.complex.Complex``
    """

    def __init__(self, molecule):
        self.molecule = molecule

    def _featurize(self):
        raise NotImplementedError("Implement this method in your subclass")

    def featurize(self):
        return self._featurize()
