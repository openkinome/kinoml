import pytest
import numpy as np
from kinoml.core.ligand import RDKitLigand
from kinoml.features.ligand import MorganFingerprintFeaturizer, OneHotSMILESFeaturizer, GraphFeaturizer
from kinoml.core.protein import Protein
from kinoml.features.protein import HashFeaturizer, AminoAcidCompositionFeaturizer, SequenceFeaturizer, GraphSeqFeaturizer

###
#  LIGAND
###

@pytest.mark.parametrize("smiles, solution", [
    ("C", np.array([0]*240 + [1] + [0]*(512-241))),
    ("B", np.array([0]*129 + [1] + [0]*(512-130))),
])
def test_ligand_MorganFingerprintFeaturizer(smiles, solution, radius=2, nbits=512):
    molecule = RDKitLigand.from_smiles(smiles)
    featurizer = MorganFingerprintFeaturizer(molecule, radius=radius, nbits=nbits)
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

@pytest.mark.parametrize("smiles, solution", [
    ("C", (np.array([[0]]), np.array([[6, 0, 0]]))),
    ("CC", (np.array([[0,1],[1,0]]), np.array([[6, 1, 1], [6, 1, 1]])))
])
def test_ligand_GraphFeaturizer(smiles, solution):
    molecule = RDKitLigand.from_smiles(smiles)
    rdmol = molecule.molecule
    featurizer = GraphFeaturizer(rdmol)
    graph = featurizer.featurize()
    assert np.array_equal(graph[1], solution[1])
    assert np.array_equal(graph[0], solution[0])



###
#  PROTEIN
###


@pytest.mark.parametrize("name, solution", [
    ("ABL1", 51359244412987742411871251883764241382133335241692098893198855723797693410968),
    ("Aàédñ", 22328255358516025959549316196054983245698711806101617436556179761578791369492)
])
def test_protein_HashFeaturizer(name, solution):
    protein = Protein(name=name)
    featurizer = HashFeaturizer(protein, normalize=False)
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


@pytest.mark.parametrize("sequence, solution", [
    ("AA", np.concatenate((np.array([[1]*2 + [0]*83]),np.zeros([19,85])), axis=0)),
    ("AAAY", np.concatenate((
        np.array([[1]*3 + [0]*82]).reshape([1, np.array([[1]*3 + [0]*82]).shape[1]]),
        np.zeros([18,85]),
        np.array([0]*3 + [1] + [0]*81).reshape([1, np.array([0]*3 + [1] + [0]*81).shape[0]])
        ), axis=0))
])
def test_protein_SequenceFeaturizer(sequence, solution):
    protein = Protein(sequence=sequence)
    featurizer = SequenceFeaturizer(protein, pad_up_to=85)
    ohe_seq = featurizer.featurize()
    assert np.array_equal(ohe_seq, solution)


@pytest.mark.parametrize("sequence, solution", [
    ("AA",(np.concatenate((np.array([[0] + [1] + [0]*83, [1] + [0]*84]),np.zeros((83,85))), axis=0),np.concatenate((np.array([[1] + [1] + [0]*83]),np.zeros((19,85))), axis=0)))
])
def test_protein_GraphSeqFeaturizer(sequence, solution):
    protein = Protein(sequence=sequence)
    featurizer = GraphSeqFeaturizer(protein, pad_up_to=85)
    graph_seq = featurizer.featurize()
    assert np.array_equal(graph_seq[0], solution[0])
    assert np.array_equal(graph_seq[1], solution[1])