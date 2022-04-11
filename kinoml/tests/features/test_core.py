"""
Test core objects of ``kinoml.features``
"""
import pytest
import numpy as np

from kinoml.core.systems import LigandSystem
from kinoml.core.ligands import Ligand
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
    TupleOfArrays,
)


def test_BaseFeaturizer():
    ligand = Ligand(smiles="CCCC")
    systems = [
        LigandSystem(components=[ligand]),
        LigandSystem(components=[ligand]),
        LigandSystem(components=[ligand]),
    ]
    featurizer = BaseFeaturizer()
    with pytest.raises(NotImplementedError):
        featurizer(systems)

    with pytest.raises(NotImplementedError):
        featurizer.featurize(systems)


def test_Pipeline():
    ligand = Ligand("CCCC")
    systems = [
        LigandSystem(components=[ligand]),
        LigandSystem(components=[ligand]),
        LigandSystem(components=[ligand]),
    ]
    featurizers = (NullFeaturizer(), NullFeaturizer())
    pipeline = Pipeline(featurizers)
    pipeline.featurize(systems)
    assert [s.featurizations["last"] for s in systems] == systems


def test_Concatenated():
    from kinoml.features.ligand import MorganFingerprintFeaturizer

    ligand = Ligand(smiles="CCCC")
    system = LigandSystem([ligand])
    featurizer1 = MorganFingerprintFeaturizer(radius=2, nbits=512, use_multiprocessing=False)
    featurizer2 = MorganFingerprintFeaturizer(radius=2, nbits=512, use_multiprocessing=False)
    concatenated = Concatenated([featurizer1, featurizer2], axis=1)
    concatenated.featurize([system])
    assert system.featurizations["last"].shape[0] == 1024


def test_TupleOfArrays():
    from kinoml.features.ligand import MorganFingerprintFeaturizer

    ligand = Ligand(smiles="CCCC")
    system = LigandSystem([ligand])
    featurizer1 = MorganFingerprintFeaturizer(radius=2, nbits=512, use_multiprocessing=False)
    featurizer2 = MorganFingerprintFeaturizer(radius=2, nbits=1024, use_multiprocessing=False)
    aggregated = TupleOfArrays([featurizer1, featurizer2])
    aggregated.featurize([system])
    assert len(system.featurizations["last"]) == 2
    assert system.featurizations["last"][0].shape[0] == 512
    assert system.featurizations["last"][1].shape[0] == 1024


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
        LigandSystem([Ligand(smiles="C")]),
        LigandSystem([Ligand(smiles="CC")]),
        LigandSystem([Ligand(smiles="CCC")]),
    )
    OneHotSMILESFeaturizer(use_multiprocessing=False).featurize(systems)
    PadFeaturizer(use_multiprocessing=False).featurize(systems)

    for s in systems:
        assert s.featurizations["last"].shape == (53, 3)

    return systems


def test_HashFeaturizer():
    system = LigandSystem([Ligand(smiles="CCC")])
    HashFeaturizer(getter=lambda s: s.ligand.molecule.to_smiles(), normalize=True).featurize(
        [system]
    )
    assert system.featurizations["last"] == pytest.approx(0.62342903)


def test_NullFeaturizer():
    system = LigandSystem([Ligand(smiles="CCC")])
    NullFeaturizer().featurize([system])

    assert system == system.featurizations["last"]


def test_CallableFeaturizer():
    from sklearn.preprocessing import scale

    systems = (
        LigandSystem([Ligand(smiles="C")]),
        LigandSystem([Ligand(smiles="CC")]),
        LigandSystem([Ligand(smiles="CCC")]),
    )
    HashFeaturizer(getter=lambda s: s.ligand.molecule.to_smiles(), normalize=False).featurize(
        systems
    )
    CallableFeaturizer(lambda s: scale(s.featurizations["last"].reshape((1,)))).featurize(systems)

    for s in systems:
        assert s.featurizations["last"].shape


def test_ClearFeaturizations_keeplast():
    from kinoml.features.ligand import OneHotSMILESFeaturizer

    systems = (
        LigandSystem([Ligand(smiles="C")]),
        LigandSystem([Ligand(smiles="CC")]),
        LigandSystem([Ligand(smiles="CCC")]),
    )
    OneHotSMILESFeaturizer(use_multiprocessing=False).featurize(systems)
    PadFeaturizer(use_multiprocessing=False).featurize(systems)
    ClearFeaturizations().featurize(systems)

    for s in systems:
        assert len(s.featurizations) == 1
        assert "last" in s.featurizations


def test_ClearFeaturizations_removeall():
    from kinoml.features.ligand import OneHotSMILESFeaturizer

    systems = (
        LigandSystem([Ligand(smiles="C")]),
        LigandSystem([Ligand(smiles="CC")]),
        LigandSystem([Ligand(smiles="CCC")]),
    )
    OneHotSMILESFeaturizer(use_multiprocessing=False).featurize(systems)
    PadFeaturizer(use_multiprocessing=False).featurize(systems)
    ClearFeaturizations(keys=tuple(), style="keep").featurize(systems)

    for s in systems:
        assert not s.featurizations
