"""
Test ligand featurizers of `kinoml.features`
"""
import pytest
import numpy as np

from kinoml.core.systems import System
from kinoml.core.ligands import Ligand
from kinoml.features.ligand import (
    SingleLigandFeaturizer,
    MorganFingerprintFeaturizer,
    OneHotSMILESFeaturizer,
    GraphLigandFeaturizer,
)


def test_single_ligand_featurizer():
    ligand1 = Ligand.from_smiles("CCCC")
    single_ligand_system = System(components=[ligand1])
    featurizer = SingleLigandFeaturizer()
    featurizer.featurize(single_ligand_system)

    ligand2 = Ligand.from_smiles("COCC")
    double_ligand_system = System(components=[ligand1, ligand2])
    with pytest.raises(ValueError):
        featurizer.featurize(double_ligand_system)


@pytest.mark.parametrize(
    "smiles, solution",
    [
        (
            "C",
            "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001",
        ),
        (
            "B",
            "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000",
        ),
    ],
)
def test_ligand_MorganFingerprintFeaturizer(smiles, solution):
    """
    OFFTK _will_ add hydrogens to all ingested SMILES, and export a canonicalized output,
    so the representation you get might not be the one you expect if you compute it directly.
    """
    ligand = Ligand.from_smiles(smiles)
    system = System([ligand])
    featurizer = MorganFingerprintFeaturizer(radius=2, nbits=512)
    fingerprint = featurizer.featurize(system)
    solution_array = np.array(list(map(int, solution)))
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
    """
    ligand = Ligand.from_smiles(smiles)
    system = System([ligand])
    featurizer = OneHotSMILESFeaturizer()
    matrix = featurizer.featurize(system)
    assert matrix.shape == solution.T.shape
    assert (matrix == solution.T).all()


@pytest.mark.parametrize(
    "smiles, solution",
    [
        ("C", (np.array([[0]]), np.array([[6, 0, 0]]))),
        ("CC", (np.array([[0, 1], [1, 0]]), np.array([[6, 1, 1], [6, 1, 1]]))),
    ],
)
def test_ligand_GraphLigandFeaturizer(smiles, solution):
    """
    OFFTK _will_ add hydrogens to all ingested SMILES, and export a canonicalized output,
    so the representation you get might not be the one you expect if you compute it directly.
    """
    ligand = Ligand.from_smiles(smiles)
    system = System([ligand])
    featurizer = GraphLigandFeaturizer()
    graph = featurizer.featurize(system)
    assert np.array_equal(graph[1], solution[1])
    assert np.array_equal(graph[0], solution[0])
