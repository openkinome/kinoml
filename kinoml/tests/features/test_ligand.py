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
def test_ligand_MorganFingerprintFeaturizer_RDKit(smiles, solution):
    """
    OFFTK _will_ add hydrogens to all ingested SMILES, and export a canonicalized output,
    so the representation you get might not be the one you expect if you compute it directly.
    """
    ligand = Ligand(smiles=smiles)
    system = LigandSystem([ligand])
    featurizer = MorganFingerprintFeaturizer(radius=2, nbits=512)
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
def test_ligand_OneHotSMILESFeaturizer_RDKit(smiles, solution):
    """
    OFFTK _will_ add hydrogens to all ingested SMILES, and export a canonicalized output,
    so the representation you get might not be the one you expect if you compute it directly.
    That's why we use RDKitLigand here.
    """
    ligand = Ligand(smiles=smiles)
    system = LigandSystem([ligand])
    featurizer = OneHotSMILESFeaturizer()
    featurizer.featurize([system])
    matrix = system.featurizations[featurizer.name]
    assert matrix.shape == solution.T.shape
    assert (matrix == solution.T).all()


@pytest.mark.parametrize(
    "smiles, solution",
    [
        (
            "C",
            (
                np.array([]),
                np.array(
                    [
                        [
                            1.0,  # start of OHE'datom.GetSymbol()
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # end of OHE'd atom.GetSymbol()
                            0.0,  # atom.GetFormalCharge()
                            0.0,  # start of OHE'd atom.GetHybridization().name
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            1.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # end of OHE'd atom.GetHybridization().name
                            0.0,  # atom.GetIsAromatic()
                            1.0,  # start of OHE'd atom.GetDegree()
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # end of OHE'd atom.GetDegree()
                            4.0,  # atom.GetTotalNumHs()
                            4.0,  # atom.GetNumImplicitHs()
                            0.0,  # atom.GetNumRadicalElectrons()
                        ]
                    ]
                ),
            ),
        ),
        (
            "CC",
            (
                np.array([[0, 1], [1, 0]]),
                np.array(
                    [
                        [  # First carbon
                            1.0,  # start of OHE'datom.GetSymbol()
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # end of OHE'd atom.GetSymbol()
                            0.0,  # atom.GetFormalCharge()
                            0.0,  # start of OHE'd atom.GetHybridization().name
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            1.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # end of OHE'd atom.GetHybridization().name
                            0.0,  # atom.GetIsAromatic()
                            0.0,  # start of OHE'd atom.GetDegree()
                            1.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # end of OHE'd atom.GetDegree()
                            3.0,  # atom.GetTotalNumHs()
                            3.0,  # atom.GetNumImplicitHs()
                            0.0,  # atom.GetNumRadicalElectrons()
                        ],
                        [  # Second carbon
                            1.0,  # start of OHE'datom.GetSymbol()
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # end of OHE'd atom.GetSymbol()
                            0.0,  # atom.GetFormalCharge()
                            0.0,  # start of OHE'd atom.GetHybridization().name
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            1.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # end of OHE'd atom.GetHybridization().name
                            0.0,  # atom.GetIsAromatic()
                            0.0,  # start of OHE'd atom.GetDegree()
                            1.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # .
                            0.0,  # end of OHE'd atom.GetDegree()
                            3.0,  # atom.GetTotalNumHs()
                            3.0,  # atom.GetNumImplicitHs()
                            0.0,  # atom.GetNumRadicalElectrons()
                        ],
                    ]
                ),
            ),
        ),
    ],
)
def test_ligand_GraphLigandFeaturizer_RDKit(smiles, solution):
    """
    OFFTK _will_ add hydrogens to all ingested SMILES, and export a canonicalized output,
    so the representation you get might not be the one you expect if you compute it directly.
    That's why we use RDKitLigand here.
    """
    ligand = Ligand(smiles=smiles)
    system = LigandSystem([ligand])
    GraphLigandFeaturizer().featurize([system])
    connectivity, features = system.featurizations["last"]
    assert (connectivity == solution[0]).all()
    assert features == pytest.approx(solution[1])
