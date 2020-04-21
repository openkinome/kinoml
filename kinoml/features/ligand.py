"""
Featurizers that mostly concern ligand-based models
"""
from __future__ import annotations
import numpy as np

from .core import BaseFeaturizer, BaseOneHotEncodingFeaturizer
from ..core.systems import System
from ..core.ligands import Ligand


class SingleLigandFeaturizer(BaseFeaturizer):
    """
    Provides a minimally useful `._supports()` method for all Ligand-like featurizers.
    """

    def _supports(self, system: System) -> bool:
        """
        Check that exactly one ligand is present in the System
        """
        super_checks = super()._supports(system)
        ligands = [c for c in system.components if isinstance(c, Ligand)]
        return all([super_checks, len(ligands) == 1])


class MorganFingerprintFeaturizer(SingleLigandFeaturizer):

    """
    Featurizes a `kinoml.core.ligand.Ligand`-like component
    using Morgan fingerprints bitvectors

    Parameters:
        radius: Morgan fingerprint neighborhood radius
        nbits: Length of the resulting bit vector
    """

    def __init__(self, radius: int = 2, nbits: int = 512, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.radius = radius
        self.nbits = nbits

    def _featurize(self, system: System) -> np.ndarray:
        """
        Featurizes a `System` as a Morgan Fingerprint using RDKit

        Returns:
            Morgan fingerprint of radius `radius` of molecule,
            with shape `nbits`.
        """
        from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as Morgan

        for component in system.components:  # we only return the first ligand found for now
            if isinstance(component, Ligand):
                mol = component.to_rdkit()
                fp = Morgan(mol, radius=self.radius, nBits=self.nbits)
                return np.asarray(fp)


class OneHotSMILESFeaturizer(BaseOneHotEncodingFeaturizer, SingleLigandFeaturizer):

    """
    One-hot encodes a `Ligand` from a canonical SMILES representation.

    Attributes:
        DICTIONARY: Defines the character-integer mapping of the one-hot encoding.
    """

    ALPHABET = (
        "BCFHIKNOPSUVWY"  # atoms
        "acegilnosru"  # aromatic atoms
        "-=#"  # bonds
        "1234567890"  # ring closures
        ".*"  # disconnections
        "()"  # branches
        "/+@:[]%\\"  # other characters
        "LR$"  # single-char representation of Cl, Br, @@
    )

    def _retrieve_sequence(self, system: System) -> str:
        """
        Get SMILES string from a `Ligand`-like component and postprocesses it.

        Double element symbols (such as `Cl`, `Br` for atoms and `@@` for chirality)
        are replaced with single element symbols (`L`, `R` and `$` respectively).
        """
        for comp in system.components:
            if isinstance(comp, Ligand):  # we only process the first one now
                return comp.to_smiles().replace("Cl", "L").replace("Br", "R").replace("@@", "$")


class OneHotRawSMILESFeaturizer(OneHotSMILESFeaturizer):
    def _retrieve_sequence(self, system: System) -> str:
        """
        Get SMILES string from a `Ligand`-like component and postprocesses it.

        Double element symbols (such as `Cl`, `Br` for atoms and `@@` for chirality)
        are replaced with single element symbols (`L`, `R` and `$` respectively).
        """
        for comp in system.components:
            if isinstance(comp, Ligand):  # we only process the first one now
                return (
                    comp._provenance["smiles"]
                    .replace("Cl", "L")
                    .replace("Br", "R")
                    .replace("@@", "$")
                )


class GraphLigandFeaturizer(SingleLigandFeaturizer):

    """
    Creates a graph representation of a `Ligand`-like component.
    Each node (atom) is decorated with several RDKit descriptors

    Check ``self._features_per_atom`` for details.

    Parameters:
        per_atom_features: function that takes a `RDKit.Chem.Atom` object
            and returns a number of features. It defaults to the internal
            `._per_atom_features` method.
    """

    def __init__(self, per_atom_features: callable = None):
        self.per_atom_features = per_atom_features or self._per_atom_features

    def _featurize(self, system: System) -> tuple:
        """
        Featurizes ligands contained in a System as a labeled graph.

        Returns:
            A two-tuple with:
            - Adjacency matrix of the molecule with shape (n_atoms, n_atoms)
            - Feature matrix with shape (n_atoms, n_features)
        """

        from rdkit import Chem
        from rdkit.Chem import rdmolops

        ligand = [c for c in system.components if isinstance(c, Ligand)][0].to_rdkit()
        adjacency_matrix = rdmolops.GetAdjacencyMatrix(ligand)
        per_atom_features = np.array([self.per_atom_features(a) for a in ligand.GetAtoms()])

        return adjacency_matrix, per_atom_features

    @staticmethod
    def _per_atom_features(atom: rdkit.Chem.Atom) -> tuple:
        """
        Computes desired features for each rdkit atom in the graph.

        Parameters:
            atom: rdkit atom to extract features from
        Returns:
            Atomic number, number of neighbors, valence
        """
        return atom.GetAtomicNum(), atom.GetDegree(), atom.GetExplicitValence()
