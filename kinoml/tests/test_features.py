####
# Ligand
####

import pytest
import numpy as np

@pytest.mark.parametrize("smiles, solution_matrix", [
    ("C", np.array([[0, 1] + [0]*51])),
    ("B", np.array([[1] + [0]*52])),
])
def test_ligand_OneHotSMILESFeaturizer(smiles, solution_matrix):
    from kinoml.features.ligand import OneHotSMILESFeaturizer
    from kinoml.core.ligand import RDKitLigand
    molecule = RDKitLigand.from_smiles(smiles)
    featurizer = OneHotSMILESFeaturizer(molecule)
    matrix = featurizer.featurize()
    assert matrix.shape == solution_matrix.T.shape
    assert (matrix == solution_matrix.T).all()