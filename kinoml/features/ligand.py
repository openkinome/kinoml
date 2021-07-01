"""
Featurizers that mostly concern ligand-based models
"""

from __future__ import annotations
from functools import lru_cache
from typing import Union, Iterable

import numpy as np
import rdkit

from .core import BaseFeaturizer, BaseOneHotEncodingFeaturizer
from ..core.systems import System
from ..core.ligands import (
    BaseLigand,
    RDKitLigand,
    SmilesLigand,
    OpenForceFieldLikeLigand,
    OpenForceFieldLigand,
)


class SingleLigandFeaturizer(BaseFeaturizer):
    """
    Provides a minimally useful ``._supports()`` method for all Ligand-like featurizers.
    """

    _COMPATIBLE_LIGAND_TYPES = (BaseLigand,)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _supports(self, system: System) -> bool:
        """
        Check that exactly one ligand is present in the System
        """
        super_checks = super()._supports(system)
        ligands = [c for c in system.components if isinstance(c, self._COMPATIBLE_LIGAND_TYPES)]
        return all([super_checks, len(ligands) == 1])

    def _find_ligand(
        self,
        system_or_ligand: Union[System, BaseLigand],
        type_=None,
    ):
        """
        Find a ligand-like component in the input object given
        to the featurizer.

        Parameters
        ----------
        system_or_ligand
            The input object to be featurized. It can be either a
            ``System``, in which case we will look for ligand-like
            objects in the components list or in the featurizations
            dictionary. It can also be a ``Ligand``-like object,
            which will be returned immediately.
        type_ : type or tuple of types, optional
            Check for specific subtypes of ligand objects. Some
            featurizers only accept Smiles, some only RDKit molecules,
            etc. By default, it will check against the class attribute
            ``_COMPATIBLE_LIGAND_TYPES``.
        """
        if type_ is None:
            type_ = self._COMPATIBLE_LIGAND_TYPES
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
    """
    Given a ``System`` containing a ``SmilesLigand`` type,
    promote it to either ``RDKitLigand`` or ``OpenForceFieldLigand``.

    Parameters
    ----------
    ligand_type : str, optional=rdkit
        If ``openforcefield``, returns ``OpenForceFieldLigand``.
        If ``rdkit``, returns ``RDKitLigand``.
    """

    _COMPATIBLE_LIGAND_TYPES = (SmilesLigand,)

    def __init__(self, ligand_type: str = "rdkit", **kwargs):
        super().__init__(**kwargs)
        if ligand_type == "rdkit":
            self._LigandType = RDKitLigand
        elif ligand_type == "openforcefield":
            self._LigandType = OpenForceFieldLigand
        else:
            raise ValueError(
                f"Ligand type `{ligand_type}` is not one of ['rkdit', 'openforcefield']"
            )

    def _featurize_one(
        self, system: Iterable[System], options: dict
    ) -> RDKitLigand | OpenForceFieldLigand:
        """
        Parameters
        ----------
        system : System
            The System to be featurized.
        options : dict
            Unused

        Returns
        -------
        ``RDKitLigand`` or ``OpenForceFieldLigand`` object
        """
        return self._LigandType.from_smiles(self._find_ligand(system).to_smiles())


class MorganFingerprintFeaturizer(SingleLigandFeaturizer):

    """
    Given a ``System`` containing one ``OpenForceFieldLikeLigand``
    component, convert it to RDKit molecule and generate
    the Morgan fingerprints bitvectors.

    Parameters
    ----------
    radius : int, optional=2
        Morgan fingerprint neighborhood radius
    nbits : int, optional=512
        Length of the resulting bit vector
    """

    _COMPATIBLE_LIGAND_TYPES = (OpenForceFieldLigand, OpenForceFieldLikeLigand)

    def __init__(self, radius: int = 2, nbits: int = 512, **kwargs):
        super().__init__(**kwargs)
        self.radius = radius
        self.nbits = nbits

    def _featurize_one(self, system: System, options: dict) -> np.ndarray:
        """
        Parameters
        ----------
        system : System
            The System to be featurized.
        options : dict
            Unused

        Returns
        -------
        array
        """
        from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as Morgan

        # FIXME: Check whether OFF uses canonical smiles internally, or not
        # otherwise, we should force that behaviour ourselves!
        ligand = self._find_ligand(system).to_rdkit()
        fp = Morgan(ligand, radius=self.radius, nBits=self.nbits)
        return np.asarray(fp, dtype="bool")


class OneHotSMILESFeaturizer(BaseOneHotEncodingFeaturizer, SingleLigandFeaturizer):

    """
    One-hot encodes a ``Ligand`` from a canonical SMILES representation.

    Attributes
    ----------
    ALPHABET : str
        Defines the character-integer mapping (as a sequence)
        of the one-hot encoding.
    """

    _COMPATIBLE_LIGAND_TYPES = (OpenForceFieldLigand, OpenForceFieldLikeLigand)
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _retrieve_sequence(self, system: System) -> str:
        """
        Get SMILES string from a `Ligand`-like component and postprocesses it.

        Double element symbols (such as `Cl`, ``Br`` for atoms and ``@@`` for chirality)
        are replaced with single element symbols (`L`, ``R`` and ``$`` respectively).
        """
        ligand = self._find_ligand(system)
        smiles = ligand.to_smiles()
        return smiles.replace("Cl", "L").replace("Br", "R").replace("@@", "$")


class OneHotRawSMILESFeaturizer(OneHotSMILESFeaturizer):
    """
    Like ``OneHotSMILESFeaturizer``, but instead of using ``ligand.to_smiles()``
    to obtain the canonical SMILES from the ligand, it relies on the stored ``metadata``
    provenance information (most possibly the original SMILES contained in the dataset).

    This should only be used for debugging purposes.
    """

    _COMPATIBLE_LIGAND_TYPES = (OpenForceFieldLigand, OpenForceFieldLikeLigand)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _retrieve_sequence(self, system: System) -> str:
        """
        Get SMILES string from a `Ligand`-like component and postprocesses it.

        Double element symbols (such as `Cl`, ``Br`` for atoms and ``@@`` for chirality)
        are replaced with single element symbols (`L`, ``R`` and ``$`` respectively).

        Parameters
        ----------
        system : System
            The system being featurized
        """
        ligand = self._find_ligand(system)
        return ligand.metadata["smiles"].replace("Cl", "L").replace("Br", "R").replace("@@", "$")


class GraphLigandFeaturizer(SingleLigandFeaturizer):

    """
    Creates a graph representation of a `Ligand`-like component.
    Each node (atom) is decorated with several RDKit descriptors
    Check ```self._per_atom_features``` for details.

    Parameters
    ----------
    max_in_ring_size : int, optional=10
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
    _COMPATIBLE_LIGAND_TYPES = (OpenForceFieldLigand, OpenForceFieldLikeLigand)

    def __init__(self, max_in_ring_size: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.max_in_ring_size = max_in_ring_size
        self._hybridization_names = sorted(rdkit.Chem.rdchem.HybridizationType.names)

    def _featurize_one(self, system: System, options: dict) -> tuple:
        """
        Featurizes ligands contained in a System as a labeled graph.

        Parameters
        ----------
        system : System
            The System being featurized
        options : dict
            Unused

        Returns
        -------
        tuple of np.array
            A two-tuple with:
            - Graph connectivity of the molecule with shape ``(2, n_edges)``
            - Feature matrix with shape ``(n_atoms, n_features)``
        """
        ligand = self._find_ligand(system).to_rdkit()
        connectivity_graph = self._connectivity_COO_format(ligand)
        # TODO: Is GetAtoms() deterministic in returned sorting?
        per_atom_features = np.array([self._per_atom_features(a) for a in ligand.GetAtoms()])

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
    def _connectivity_COO_format(mol: rdkit.Chem.Mol) -> np.ndarray:
        """
        Returns the connectivity of the molecular graph in COO format.

        Parameters
        ----------
        mol: rdkit.Chem.Mol
            rdkit molecule to extract bonds from

        Returns
        -------
        np.ndarray
            graph connectivity in COO format with shape ``[2, num_edges]``
        """

        row, col = [], []

        # TODO: Is GetBonds() deterministic?
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]

        return np.array([row, col])
