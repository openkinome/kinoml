import pytest
import numpy as np
from kinoml.core.ligand import RDKitLigand
from kinoml.features.ligand import OneHotSMILESFeaturizer, MorganFingerprintFeaturizer
from kinoml.core.protein import Protein
from kinoml.features.protein import HashFeaturizer, AminoAcidCompositionFeaturizer

###
#  LIGAND
###

@pytest.mark.parametrize("smiles, solution", [
    ("C", np.array([0]*240 + [1] + [0]*(512-241))),
    ("B", np.array([0]*129 + [1] + [0]*(512-130))),
])
def test_ligand_MorganFingerprintFeaturizer(smiles, solution, radius=2, nbits=512):
    molecule = RDKitLigand.from_smiles(smiles)
    rdmol = molecule.molecule
    featurizer = MorganFingerprintFeaturizer(rdmol, radius=radius, nbits=nbits)
    fingerprint = featurizer.featurize()
    assert (fingerprint == solution).all()


@pytest.mark.parametrize("smiles, solution", [
    ("C", np.array([[0, 1] + [0]*51])),
    ("B", np.array([[1] + [0]*52])),
    ("CC", np.array([[0, 1] + [0]*51,
                     [0, 1] + [0]*51]))
])
def test_ligand_OneHotSMILESFeaturizer(smiles, solution):

    molecule = RDKitLigand.from_smiles(smiles)
    featurizer = OneHotSMILESFeaturizer(molecule)
    matrix = featurizer.featurize()
    assert matrix.shape == solution.T.shape
    assert (matrix == solution.T).all()


###
#  PROTEIN
###


@pytest.mark.parametrize("name, solution", [
    ("ABL1", 51359244412987742411871251883764241382133335241692098893198855723797693410968),
    ("TK", 94240903631043095486486222526930572378578908819371024688233288433881997985526),
    ("Ññ", 80345342214904496436064957164160100315589462262335309353169746645107183191772)
])
def test_protein_HashFeaturizer(name, solution):
    protein = Protein(name=name)
    featurizer = HashFeaturizer(protein)
    hashed = featurizer.featurize()
    assert hashed == solution


@pytest.mark.parametrize("sequence, solution", [
    ("AA", np.array([2]*1 + [0]*19)),
    ("KKLGAGQFGEVWMVAVKTMAFLAEANVMKTLQDKLVKLHAVYIITEFMAKGSLLDFLKSFIEQRNYIHRDLRAANILVIADFGLA",
    np.array([11, 0, 4, 4, 6, 5, 2, 6, 8, 11, 4, 3, 0, 3, 3, 2, 3, 7, 1, 2]))
])
def test_protein_AminoAcidCompositionFeaturizer(sequence, solution):
    protein = Protein(sequence=sequence)
    featurizer = AminoAcidCompositionFeaturizer(protein)
    composition = featurizer.featurize()
    assert (composition == solution).all()
