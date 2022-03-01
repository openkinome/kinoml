"""
Featurizers that mostly concern ligand-based models
"""

from __future__ import annotations
from typing import Union

import numpy as np
from openff.toolkit.utils.exceptions import SMILESParseError
from rdkit import Chem

from .core import ParallelBaseFeaturizer, BaseOneHotEncodingFeaturizer
from ..core.systems import LigandSystem, ProteinLigandComplex
from ..core.ligands import Ligand


class SingleLigandFeaturizer(ParallelBaseFeaturizer):
    """
    Provides a minimally useful ``._supports()`` method for all Ligand-like featurizers.
    """

    _COMPATIBLE_LIGAND_TYPES = (Ligand,)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _supports(self, system: Union[LigandSystem, ProteinLigandComplex]) -> bool:
        """
        Check that exactly one ligand is present in the System
        """
        super_checks = super()._supports(system)
        ligands = [c for c in system.components if isinstance(c, self._COMPATIBLE_LIGAND_TYPES)]
        return all([super_checks, len(ligands) == 1])


class MorganFingerprintFeaturizer(SingleLigandFeaturizer):
    """
    Given a ``System`` containing one ``Ligand`` component, convert it to an RDKit molecule and
    generate the Morgan fingerprints bitvectors.

    Parameters
    ----------
    radius: int, optional=2
        Morgan fingerprint neighborhood radius
    nbits: int, optional=512
        Length of the resulting bit vector
    """

    def __init__(self, radius: int = 2, nbits: int = 512, **kwargs):
        super().__init__(**kwargs)
        self.radius = radius
        self.nbits = nbits

    def _featurize_one(
        self, system: Union[LigandSystem, ProteinLigandComplex]
    ) -> Union[np.ndarray, None]:
        """
        Return the Morgan fingerprint for the given system.

        Parameters
        ----------
        system: LigandSystem or ProteinLigandComplex
            The System to be featurized.

        Returns
        -------
            : np.array or None
        """
        from rdkit.Chem import RemoveHs
        from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

        try:  # catch erroneous smiles not yet interpreted in case of lazy instantiation
            rdkit_mol = system.ligand.molecule.to_rdkit()
        except SMILESParseError:
            return None

        rdkit_mol = RemoveHs(rdkit_mol)
        fp = GetMorganFingerprintAsBitVect(rdkit_mol, radius=self.radius, nBits=self.nbits)
        return np.asarray(fp, dtype="int64")


class OneHotSMILESFeaturizer(BaseOneHotEncodingFeaturizer, SingleLigandFeaturizer):

    """
    One-hot encodes a ``Ligand`` from a SMILES representation.

    Attributes
    ----------
    ALPHABET: str
        Defines the character-integer mapping (as a sequence)
        of the one-hot encoding.
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

    def __init__(self, smiles_type: str = "canonical", **kwargs):
        """
        One-hot encodes a ``Ligand`` from a SMILES representation.

        Parameters
        ----------
        smiles_type: str, default=canonical
            The smiles type to use ('canonical' or 'raw').
        """
        super().__init__(**kwargs)
        if smiles_type not in ["canonical", "raw"]:
            raise ValueError(
                "Only 'canonical' and 'raw' are supported smiles_type, you provided "
                f"{smiles_type}."
            )
        self.smiles_type = smiles_type

    def _retrieve_sequence(self, system: Union[LigandSystem, ProteinLigandComplex]) -> str:
        """
        Get SMILES string from a `Ligand`-like component and postprocesses it.

        Double element symbols (such as `Cl`, ``Br`` for atoms and ``@@`` for chirality)
        are replaced with single element symbols (`L`, ``R`` and ``$`` respectively).
        """
        try:
            if self.smiles_type == "canonical":
                smiles = system.ligand.molecule.to_smiles(explicit_hydrogens=False)
            else:
                smiles = system.ligand.metadata["smiles"]
        except SMILESParseError:  # erroneous SMILES string
            return ""
        except KeyError:  # no SMILES string given during initialization
            return ""

        return smiles.replace("Cl", "L").replace("Br", "R").replace("@@", "$")


class GraphLigandFeaturizer(SingleLigandFeaturizer):

    """
    Creates a graph representation of a `Ligand`-like component.
    Each node (atom) is decorated with several RDKit descriptors
    Check ```self._per_atom_features``` for details.

    Parameters
    ----------
    max_in_ring_size: int, optional=10
        Maximum ring size for testing whether an atom belongs to a
        ring or not. *Currently unused*
    """

    ALL_ATOMIC_SYMBOLS = [
        "C",
        "N",
        "O",
        "S",
        "F",
        "Si",
        "P",
        "Cl",
        "Br",
        "Mg",
        "Na",
        "Ca",
        "Fe",
        "As",
        "Al",
        "I",
        "B",
        "V",
        "K",
        "Tl",
        "Yb",
        "Sb",
        "Sn",
        "Ag",
        "Pd",
        "Co",
        "Se",
        "Ti",
        "Zn",
        "H",
        "Li",
        "Ge",
        "Cu",
        "Au",
        "Ni",
        "Cd",
        "In",
        "Mn",
        "Zr",
        "Cr",
        "Pt",
        "Hg",
        "Pb",
        "Unknown",
    ]

    def __init__(self, max_in_ring_size: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.max_in_ring_size = max_in_ring_size
        self._hybridization_names = sorted(Chem.rdchem.HybridizationType.names)

    def _featurize_one(
        self, system: Union[LigandSystem, ProteinLigandComplex]
    ) -> Union[tuple, None]:
        """
        Featurizes ligands contained in a System as a labeled graph.

        Parameters
        ----------
        system: LigandSystem or ProteinLigandComplex
            The System being featurized.

        Returns
        -------
        tuple of np.array or None
            A two-tuple with:
            - Graph connectivity of the molecule with shape ``(2, n_edges)``
            - Feature matrix with shape ``(n_atoms, n_features)``
        """
        try:  # catch erroneous smiles not yet interpreted in case of lazy instantiation
            # rdkit_mol = system.ligand.molecule.to_rdkit()
            # this does not work, since openff toolkit will permit implicit hydrogens when
            # converting to rdkit (see https://github.com/openforcefield/openff-toolkit/pull/1001)
            smiles = system.ligand.molecule.to_smiles(explicit_hydrogens=False)
            rdkit_mol = Chem.MolFromSmiles(smiles)
        except SMILESParseError:
            return None

        connectivity_graph = self._connectivity_COO_format(rdkit_mol)
        per_atom_features = np.array([self._per_atom_features(a) for a in rdkit_mol.GetAtoms()])
        return connectivity_graph, per_atom_features

    def _per_atom_features(self, atom) -> np.ndarray:
        """
        Computes desired features for each atom in the molecular graph.

        Parameters
        ----------
        atom: rdkit.Chem.Atom
            Atom to extract features from

        Returns
        -------
        tuple of atomic features.
            atomic_symbol : array
                the one-hot encoded atomic symbol from `ALL_ATOMIC_SYMBOLS`.
            formal_charge : int
                the formal charge of atom.
            hybridization_type : array
                the one-hot encoded hybridization type from
                ``rdkit.Chem.rdchem.HybridizationType``.
            aromatic : bool
                if atom is aromatic.
            degree : array
                the one-hot encoded degree of the atom in the molecule.
            total_h : int
                the total number of hydrogens on the atom (implicit and explicit).
            implicit_h : int
                the number of implicit hydrogens on the atom.
            radical_electrons : int
                the number of radical electrons.

        Notes
        -----
        The atomic features are the same as in PotentialNet [1]_.

        .. [1] https://doi.org/10.1021/acscentsci.8b00507
        """
        # Return flattened array; notice how the OHE'd matrices are flattened
        # and iterated with the * unpacking operator --
        return np.array(
            [
                # 1. Chemical element, one-hot encoded
                *BaseOneHotEncodingFeaturizer.one_hot_encode(
                    [atom.GetSymbol()], self.ALL_ATOMIC_SYMBOLS
                ).flatten(),
                # 2. Formal charge
                atom.GetFormalCharge(),
                # 3. Hybridization, one-hot encoded
                *BaseOneHotEncodingFeaturizer.one_hot_encode(
                    [atom.GetHybridization().name],
                    self._hybridization_names,
                ).flatten(),
                # 4. Aromaticity
                atom.GetIsAromatic(),
                # 5. Total numbers of bonds, one-hot encoded
                *BaseOneHotEncodingFeaturizer.one_hot_encode(
                    [atom.GetDegree()], list(range(11))
                ).flatten(),
                # 6. Total number of hydrogens
                atom.GetTotalNumHs(),
                # 7. Number of implicit hydrogens
                atom.GetNumImplicitHs(),
                # 8. Number of radical electrons
                atom.GetNumRadicalElectrons(),
            ],
            dtype="float64",
        )

    @staticmethod
    def _connectivity_COO_format(mol: Chem.Mol) -> np.ndarray:
        """
        Returns the connectivity of the molecular graph in COO format.

        Parameters
        ----------
        mol: rdkit.Chem.Mol
            RDKit molecule to extract bonds from

        Returns
        -------
        np.ndarray
            graph connectivity in COO format with shape ``[2, num_edges]``
        """

        row, col = [], []

        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]

        return np.array([row, col])
