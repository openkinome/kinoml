"""
Test ligand featurizers of `kinoml.features`
"""
import pytest
import numpy as np

from kinoml.core.systems import System
from kinoml.core.ligands import Ligand, RDKitLigand
from kinoml.features.ligand import (
    SingleLigandFeaturizer,
    MorganFingerprintFeaturizer,
    OneHotSMILESFeaturizer,
    GraphLigandFeaturizer,
)


@pytest.mark.parametrize("LigandClass", [Ligand, RDKitLigand])
def test_single_ligand_featurizer(LigandClass):
    ligand1 = LigandClass.from_smiles("CCCC")
    single_ligand_system = System(components=[ligand1])
    featurizer = SingleLigandFeaturizer()
    featurizer.supports(single_ligand_system)

    ligand2 = Ligand.from_smiles("COCC")
    double_ligand_system = System(components=[ligand1, ligand2])
    with pytest.raises(ValueError):
        featurizer.featurize(double_ligand_system)


@pytest.mark.parametrize(
    "smiles, solution",
    [
        (
            "C",
            "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        ),
        (
            "B",
            "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        ),
    ],
)
def test_ligand_MorganFingerprintFeaturizer(smiles, solution):
    """
    OFFTK _will_ add hydrogens to all ingested SMILES, and export a canonicalized output,
    so the representation you get might not be the one you expect if you compute it directly.
    """
    ligand = RDKitLigand.from_smiles(smiles)
    system = System([ligand])
    featurizer = MorganFingerprintFeaturizer(radius=2, nbits=512)
    featurizer.featurize(system)
    fingerprint = system.featurizations[featurizer.name]
    solution_array = np.array(list(map(int, solution)), dtype="uint8")
    assert (fingerprint == solution_array).all()


@pytest.mark.parametrize(
    "smiles, solution",
    [
        ("C", np.array([[0, 1] + [0] * 51])),
        ("B", np.array([[1] + [0] * 52])),
        ("CC", np.array([[0, 1] + [0] * 51, [0, 1] + [0] * 51])),
    ],
)
def test_ligand_OneHotSMILESFeaturizer(smiles, solution):
    """
    OFFTK _will_ add hydrogens to all ingested SMILES, and export a canonicalized output,
    so the representation you get might not be the one you expect if you compute it directly.
    That's why we use RDKitLigand here.
    """
    ligand = RDKitLigand.from_smiles(smiles)
    system = System([ligand])
    featurizer = OneHotSMILESFeaturizer()
    featurizer.featurize(system)
    matrix = system.featurizations[featurizer.name]
    assert matrix.shape == solution.T.shape
    assert (matrix == solution.T).all()


@pytest.mark.parametrize(
    "smiles, solution",
    [
        (
            "C",
            (
                np.array([[0]]),
                [(6, "C", 0, 4, 0, 4, 4, 12.011, 0, 0, 4, 4, False, 0, False, 0, 4)],
            ),
        ),
        (
            "CC",
            (
                np.array([[0, 1], [1, 0]]),
                np.array(
                    [
                        (6, "C", 1, 4, 1, 3, 4, 12.011, 0, 0, 3, 3, False, 0, False, 0, 4),
                        (6, "C", 1, 4, 1, 3, 4, 12.011, 0, 0, 3, 3, False, 0, False, 0, 4),
                    ]
                ),
            ),
        ),
    ],
)
def test_ligand_GraphLigandFeaturizer(smiles, solution):
    """
    OFFTK _will_ add hydrogens to all ingested SMILES, and export a canonicalized output,
    so the representation you get might not be the one you expect if you compute it directly.
    That's why we use RDKitLigand here.
    """
    ligand = RDKitLigand.from_smiles(smiles)
    system = System([ligand])
    featurizer = GraphLigandFeaturizer()
    featurizer.featurize(system)
    graph = system.featurizations[featurizer.name]
    assert (graph[0] == solution[0]).all()  # connectivity
    assert (graph[1] == solution[1]).all()  # features
