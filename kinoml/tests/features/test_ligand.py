"""
Test ligand featurizers of `kinoml.features`
"""
import pytest
import numpy as np

from kinoml.core.systems import LigandSystem
from kinoml.core.ligands import Ligand
from kinoml.features.ligand import (
    SingleLigandFeaturizer,
    MorganFingerprintFeaturizer,
    OneHotSMILESFeaturizer,
    GraphLigandFeaturizer,
)


def test_single_ligand_featurizer():
    ligand1 = Ligand(smiles="CCCC")
    single_ligand_system = LigandSystem(components=[ligand1])
    featurizer = SingleLigandFeaturizer()
    featurizer.supports(single_ligand_system)

    ligand2 = Ligand(smiles="COCC")
    double_ligand_system = LigandSystem(components=[ligand1, ligand2])
    with pytest.raises(ValueError):
        featurizer.featurize([double_ligand_system])


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
    ligand = Ligand(smiles=smiles)
    system = LigandSystem([ligand])
    featurizer = MorganFingerprintFeaturizer(radius=2, nbits=512, use_multiprocessing=False)
    featurizer.featurize([system])
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
    ligand = Ligand(smiles=smiles)
    system = LigandSystem([ligand])
    featurizer = OneHotSMILESFeaturizer(use_multiprocessing=False)
    featurizer.featurize([system])
    matrix = system.featurizations[featurizer.name]
    assert matrix.shape == solution.T.shape
    assert (matrix == solution.T).all()


@pytest.mark.parametrize(
    "smiles, n_edges, n_nodes, n_features",
    [("C", 0, 1, 69), ("CC", 2, 2, 69)],
)
def test_ligand_GraphLigandFeaturizer_RDKit(smiles, n_edges, n_nodes, n_features):
    ligand = Ligand(smiles=smiles)
    system = LigandSystem([ligand])
    GraphLigandFeaturizer(use_multiprocessing=False).featurize([system])
    connectivity, features = system.featurizations["last"]
    assert len(connectivity) == n_edges
    assert len(features[0]) == n_nodes
    assert len(features[1]) == n_features
