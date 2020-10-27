"""
Featurizers that mostly concern ligand-based models
"""
from __future__ import annotations
import numpy as np
from functools import lru_cache

from .core import BaseFeaturizer, BaseOneHotEncodingFeaturizer
from ..core.systems import System
from ..core.ligands import BaseLigand, SmilesLigand, Ligand


class SingleLigandFeaturizer(BaseFeaturizer):
    """
    Provides a minimally useful `._supports()` method for all Ligand-like featurizers.
    """

    def _supports(self, system: System) -> bool:
        """
        Check that exactly one ligand is present in the System
        """
        super_checks = super()._supports(system)
        ligands = [c for c in system.components if isinstance(c, BaseLigand)]
        return all([super_checks, len(ligands) == 1])

    def _find_ligand(self, system_or_ligand: Union[System, BaseLigand], type_=Ligand):
        if isinstance(system_or_ligand, type_):
            return system_or_ligand
        # we only return the first ligand found for now
        for component in system_or_ligand.components:
            if isinstance(component, type_):
                ligand = component
                break
        else:  # look in featurizations?
            for key, feature in system_or_ligand.featurizations.items():
                if isinstance(feature, type_):
                    ligand = feature
                    break
            else:
                raise ValueError(f"No {type_} instances found in system {system_or_ligand}")
        return ligand


class SmilesToLigandFeaturizer(SingleLigandFeaturizer):
    _styles = ("openforcefield", "rdkit")

    def __init__(self, style="openforcefield"):
        assert (
            style in self._styles
        ), f"`{self.__class__.__name__}.style` must be one of {self._styles}"
        self._style = style

    def _supports(self, system):
        super_checks = super()._supports(system)
        ligands = [c for c in system.components if isinstance(c, SmilesLigand)]
        return all([super_checks, len(ligands) == 1])

    @lru_cache(maxsize=1000)
    def _featurize(self, system: System) -> np.ndarray:
        """
        Featurizes a `SmilesLigand` component and builds a `Ligand` object

        Returns:
            `Ligand` object
        """
        ligand = self._find_ligand(system, type_=SmilesLigand)
        if self._style == "openforcefield":
            return Ligand.from_smiles(ligand.smiles, name=ligand.name)
        elif self._style == "rdkit":
            from rdkit.Chem import MolFromSmiles

            return MolFromSmiles(ligand.smiles)
        else:
            raise ValueError(f"`{self.__class__.__name__}.style` must be one of {self._styles}")


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
        from rdkit.Chem import Mol as RDKitMol

        ligand = self._find_ligand(system, type_=(Ligand, RDKitMol))
        return self._featurize_ligand(ligand)

    @lru_cache(maxsize=1000)
    def _featurize_ligand(self, ligand):
        from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as Morgan

        # FIXME: Check whether OFF uses canonical smiles internally, or not
        # otherwise, we should force that behaviour ourselves!
        if isinstance(ligand, Ligand):
            ligand = ligand.to_rdkit()
        fp = Morgan(ligand, radius=self.radius, nBits=self.nbits)
        return np.asarray(fp, dtype="uint8")


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
        ligand = self._find_ligand(system)
        return ligand.to_smiles().replace("Cl", "L").replace("Br", "R").replace("@@", "$")


class OneHotRawSMILESFeaturizer(OneHotSMILESFeaturizer):
    def _retrieve_sequence(self, system: System) -> str:
        """
        Get SMILES string from a `Ligand`-like component and postprocesses it.

        Double element symbols (such as `Cl`, `Br` for atoms and `@@` for chirality)
        are replaced with single element symbols (`L`, `R` and `$` respectively).
        """
        ligand = self._find_ligand(system)
        return ligand.metadata["smiles"].replace("Cl", "L").replace("Br", "R").replace("@@", "$")


class GraphLigandFeaturizer(SingleLigandFeaturizer):

    """
    Creates a graph representation of a `Ligand`-like component.
    Each node (atom) is decorated with several RDKit descriptors

    Check ``self._per_atom_features`` for details.

    Parameters:
        per_atom_features: function that takes a `RDKit.Chem.Atom` object
            and returns a number of features. It defaults to the internal
            `._per_atom_features` method.
    """

    def __init__(self, per_atom_features: callable = None):
        self.per_atom_features = per_atom_features or self._per_atom_features

    @lru_cache(maxsize=1000)
    def _featurize(self, system: System) -> tuple:
        """
        Featurizes ligands contained in a System as a labeled graph.

        Returns:
            A two-tuple with:
            - Graph connectivity of the molecule with shape (2, n_edges)
            - Feature matrix with shape (n_atoms, n_features)
        """

        from rdkit import Chem

        ligand = self._find_ligand(system).to_rdkit()
        connectivity_graph = self._connectivity_COO_format(ligand)
        per_atom_features = np.array([self._per_atom_features(a) for a in ligand.GetAtoms()])

        return connectivity_graph, per_atom_features

    @staticmethod
    def _per_atom_features(atom: rdkit.Chem.Atom) -> tuple:
        """
        Computes desired features for each rdkit atom in the graph.

        Parameters:
            atom: rdkit atom to extract features from
        Returns:
            Atomic number, number of neighbors, valence,
            atomic mass, formal charge, number of implicit hydrogens,
            bool (if atom is in a ring), bool (if atom is aromatic), number of radical electrons
        """
        return (
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetExplicitValence(),
            atom.GetMass(),
            atom.GetFormalCharge(),
            atom.GetNumImplicitHs(),
            atom.IsInRing(),
            atom.GetIsAromatic(),
            atom.GetNumRadicalElectrons(),
        )

    @staticmethod
    def _connectivity_COO_format(mol: rdkit.Chem.rdchem.Mol) -> array:
        """
        Returns the connectivity of the molecular graph in COO format.

        Parameters:
            mol: rdkit molecule to extract bonds from
        Returns:
            array: graph connectivity in COO format with shape [2, num_edges]
        """

        row, col = [], []

        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]

        return np.array([row, col])
