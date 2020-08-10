"""
Featurizers that can only get applied to ProteinLigandComplexes or
subclasses thereof
"""
from __future__ import annotations
from functools import lru_cache

from .core import BaseFeaturizer
from ..core.systems import ProteinLigandComplex


class OpenEyesProteinLigandDockingFeaturizer(BaseFeaturizer):

    """
    Given a System with exactly one protein and one ligand,
    dock the ligand in the designated binding pocket.

    We assume that a file-based System object will be passed; this
    means we will have a System.components with (FileProtein, FileLigand).
    The file itself could be a URL.
    """

    _SUPPORTED_TYPES = (ProteinLigandComplex,)

    @lru_cache(maxsize=100)
    def _featurize(self, system: ProteinLigandComplex) -> ProteinLigandComplex:
        import oedocking

        # add your stuff here, using as many supporting self._methods() as needed!
