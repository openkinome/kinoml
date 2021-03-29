"""
Featurizers that mostly concern ligand-based models
"""
from __future__ import annotations
from functools import lru_cache

import numpy as np
import rdkit

from .core import BaseFeaturizer, BaseOneHotEncodingFeaturizer
from ..core.systems import System
from ..core.ligands import BaseLigand, SmilesLigand, Ligand

ALL_ATOMIC_SYMBOLS = [
    'C',
    'N',
    'O',
    'S',
    'F',
    'Si',
    'P',
    'Cl',
    'Br',
    'Mg',
    'Na',
    'Ca',
    'Fe',
    'As',
    'Al',
    'I',
    'B',
    'V',
    'K',
    'Tl',
    'Yb',
    'Sb',
    'Sn',
    'Ag',
    'Pd',
    'Co',
    'Se',
    'Ti',
    'Zn',
    'H',
    'Li',
    'Ge',
    'Cu',
    'Au',
    'Ni',
    'Cd',
    'In',
    'Mn',
    'Zr',
    'Cr',
    'Pt',
    'Hg',
    'Pb',
    'Unknown'
]

SIZE_OF_RING = [3, 4, 5, 6, 7, 8, 9, 10]

HYBRIZIDATION_TYPES = [
    rdkit.Chem.rdchem.HybridizationType.OTHER, # OTHER
    rdkit.Chem.rdchem.HybridizationType.S, # S
    rdkit.Chem.rdchem.HybridizationType.SP, # SP
    rdkit.Chem.rdchem.HybridizationType.SP2, # SP2
    rdkit.Chem.rdchem.HybridizationType.SP3, # SP2
    rdkit.Chem.rdchem.HybridizationType.SP3D, # SP3D
    rdkit.Chem.rdchem.HybridizationType.SP3D2, # SP3D2
    rdkit.Chem.rdchem.HybridizationType.UNSPECIFIED # UNSPECIFIED
]

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
            for feature in system_or_ligand.featurizations.values():
                if isinstance(feature, type_):
                    ligand = feature
                    break
            else:
                raise ValueError(f"No {type_} instances found in system {system_or_ligand}")
        return ligand


class SmilesToLigandFeaturizer(SingleLigandFeaturizer):
    def __init__(self, style="openforcefield"):
        self._style = style
        self._build_ligand = getattr(self, f"_build_ligand_{style}")

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
        return self._build_ligand(ligand)

    def _build_ligand_openforcefield(self, ligand):
        return Ligand.from_smiles(ligand.smiles, allow_undefined_stereo=True)

    def _build_ligand_rdkit(self, ligand):
        from rdkit.Chem import MolFromSmiles

        return MolFromSmiles(ligand.smiles)


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

    def __init__(self, style="openforcefield", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._get_smiles = getattr(self, f"_get_smiles_{style}")

        # TODO: Add canonicalization

    def _retrieve_sequence(self, system: System) -> str:
        """
        Get SMILES string from a `Ligand`-like component and postprocesses it.

        Double element symbols (such as `Cl`, `Br` for atoms and `@@` for chirality)
        are replaced with single element symbols (`L`, `R` and `$` respectively).
        """
        ligand = self._find_ligand(system)
        smiles = self._get_smiles(ligand)
        return smiles.replace("Cl", "L").replace("Br", "R").replace("@@", "$")

    def _get_smiles_openforcefield(self, ligand):
        return ligand.to_smiles()

    def _get_smiles_rdkit(self, ligand):
        from rdkit.Chem import MolToSmiles

        return MolToSmiles(ligand)

    def _get_smiles_metadata(self, ligand):
        return ligand.metadata["smiles"]


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
    def _per_atom_features(atom):
        """
        Computes desired features for each atom in the molecular graph.

        Parameters
        ----------
            atom: atom to extract features from

        Returns
        -------
        tuple of atomic features (all 17 included by default).

            atomic_number : int
                the atomic number.
            atomic_symbol : array
                the one-hot encoded atomic symbol from `ALL_ATOMIC_SYMBOLS`.
            degree : int
                the degree of the atom in the molecule (number of neighbors).
            total_degree : int
                the degree of the atom in the molecule including hydrogens.
            explicit_valence : int
                the explicit valence of the atom.
            implicit_valence : int
                the number of implicit Hs on the atom.
            total_valence : int
                the total valence (explicit + implicit) of the atom.
            atomic_mass : float
                the atomic mass.
            formal_charge : int
                the formal charge of atom.
            explicit_h : int
                the number of explicit hydrogens.
            implicit_h : int
                the total number of implicit hydrogens on the atom.
            total_h : int
                the total number of Hs (explicit and implicit) on the atom.
            ring : bool
                if the atom is part of a ring.
            ring_size : array
                if the atom if part of a ring of size determined by `SIZE_OF_RING`.
            aromatic : bool
                    if atom is aromatic
            radical_electrons : int
                number of radical electrons
            hybridization_type : array
                the one-hot enocded hybridization type from `HYBRIZIDATION_TYPES`.
        """
        return (
            atom.GetAtomicNum(),
            atom.GetSymbol(), # TODO : one-hot encode
            atom.GetDegree(),
            atom.GetTotalDegree(), # TODO : ignore if molecule has H
            atom.GetExplicitValence(),
            atom.GetImplicitValence(), # TODO : ignore if molecule has H
            atom.GetTotalValence(),
            atom.GetMass(),
            atom.GetFormalCharge(),
            atom.GetNumExplicitHs(),
            atom.GetNumImplicitHs(),
            atom.GetTotalNumHs(),
            atom.IsInRing(),
            atom.IsInRingSize(), # TODO : one-hot encode
            atom.GetIsAromatic(),
            atom.GetNumRadicalElectrons(),
            atom.GetHybridization(), # TODO : one-hot encode
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
