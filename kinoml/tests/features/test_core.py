"""
Test core objects of ``kinoml.features``
"""
import pytest
import numpy as np

from kinoml.core.systems import System, LigandSystem
from kinoml.core.ligands import RDKitLigand, SmilesLigand
from kinoml.features.core import (
    BaseFeaturizer,
    Pipeline,
    Concatenated,
    BaseOneHotEncodingFeaturizer,
    PadFeaturizer,
    HashFeaturizer,
    NullFeaturizer,
    CallableFeaturizer,
    ClearFeaturizations,
)


def test_BaseFeaturizer():
    ligand = SmilesLigand.from_smiles("CCCC")
    systems = System(components=[ligand]), System(components=[ligand]), System(components=[ligand])
    featurizer = BaseFeaturizer()
    with pytest.raises(NotImplementedError):
        featurizer(systems)

    with pytest.raises(NotImplementedError):
        featurizer.featurize(systems)


def test_Pipeline():
    ligand = SmilesLigand.from_smiles("CCCC")
    systems = System(components=[ligand]), System(components=[ligand]), System(components=[ligand])
    featurizers = (NullFeaturizer(), NullFeaturizer())
    pipeline = Pipeline(featurizers)
    assert pipeline.featurize(systems) == systems


def test_Concatenated():
    from kinoml.features.ligand import MorganFingerprintFeaturizer

    ligand = RDKitLigand.from_smiles("CCCC")
    system = System([ligand])
    featurizer1 = MorganFingerprintFeaturizer(radius=2, nbits=512)
    featurizer2 = MorganFingerprintFeaturizer(radius=2, nbits=512)
    concatenated = Concatenated([featurizer1, featurizer2], axis=1)
    concatenated.featurize([system])
    assert system.featurizations["last"].shape[0] == 1024


def test_BaseOneHotEncodingFeaturizer():
    assert (
        BaseOneHotEncodingFeaturizer.one_hot_encode("AAA", "ABC") == np.array([[1, 0, 0]] * 3).T
    ).all()
    assert (
        BaseOneHotEncodingFeaturizer.one_hot_encode("AAA", {"A": 0, "B": 1, "C": 2})
        == np.array([[1, 0, 0]] * 3).T
    ).all()
    assert (
        BaseOneHotEncodingFeaturizer.one_hot_encode(["A", "A", "A"], ["A", "B", "C"])
        == np.array([[1, 0, 0]] * 3).T
    ).all()


def test_PadFeaturizer():
    from kinoml.features.ligand import OneHotSMILESFeaturizer

    systems = (
        System([RDKitLigand.from_smiles("C")]),
        System([RDKitLigand.from_smiles("CC")]),
        System([RDKitLigand.from_smiles("CCC")]),
    )
    OneHotSMILESFeaturizer().featurize(systems)
    PadFeaturizer().featurize(systems)

    for s in systems:
        assert s.featurizations["last"].shape == (53, 3)

    return systems


def test_HashFeaturizer():
    system = LigandSystem([SmilesLigand.from_smiles("CCC")])
    HashFeaturizer(getter=lambda s: s.ligand.to_smiles(), normalize=True).featurize([system])
    assert system.featurizations["last"] == pytest.approx(0.54818723)


def test_NullFeaturizer():
    system = LigandSystem([SmilesLigand.from_smiles("CCC")])
    NullFeaturizer().featurize([system])

    assert system == system.featurizations["last"]


def test_CallableFeaturizer():
    from sklearn.preprocessing import scale

    systems = (
        LigandSystem([RDKitLigand.from_smiles("C")]),
        LigandSystem([RDKitLigand.from_smiles("CC")]),
        LigandSystem([RDKitLigand.from_smiles("CCC")]),
    )
    HashFeaturizer(getter=lambda s: s.ligand.to_smiles(), normalize=False).featurize(systems)
    CallableFeaturizer(lambda s: scale(s.featurizations["last"].reshape((1,)))).featurize(systems)

    for s in systems:
        assert s.featurizations["last"].shape


def test_ClearFeaturizations_keeplast():
    from kinoml.features.ligand import OneHotSMILESFeaturizer

    systems = (
        System([RDKitLigand.from_smiles("C")]),
        System([RDKitLigand.from_smiles("CC")]),
        System([RDKitLigand.from_smiles("CCC")]),
    )
    OneHotSMILESFeaturizer().featurize(systems)
    PadFeaturizer().featurize(systems)
    ClearFeaturizations().featurize(systems)

    for s in systems:
        assert len(s.featurizations) == 1
        assert "last" in s.featurizations


def test_ClearFeaturizations_removeall():
    from kinoml.features.ligand import OneHotSMILESFeaturizer

    systems = (
        System([RDKitLigand.from_smiles("C")]),
        System([RDKitLigand.from_smiles("CC")]),
        System([RDKitLigand.from_smiles("CCC")]),
    )
    OneHotSMILESFeaturizer().featurize(systems)
    PadFeaturizer().featurize(systems)
    ClearFeaturizations(keys=tuple(), style="keep").featurize(systems)

    for s in systems:
        assert not s.featurizations
