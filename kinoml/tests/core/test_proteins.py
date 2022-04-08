"""
Test kinoml.core.proteins
"""
from importlib import resources

from MDAnalysis.core.universe import Universe
from openeye import oechem
import pandas as pd


def test_protein_from_file():
    """Check from file reading with MDAnalysis and OpenEye."""
    from kinoml.core.proteins import Protein

    with resources.path("kinoml.data.proteins", "4f8o.pdb") as path:
        protein = Protein.from_file(path)
        assert isinstance(protein.molecule, oechem.OEGraphMol)
        protein = Protein.from_file(path, toolkit="MDAnalysis")
        assert isinstance(protein.molecule, Universe)


def test_protein_from_pdb():
    """Check instantation from PDB ID."""
    from kinoml.core.proteins import Protein

    protein = Protein.from_pdb("4yne")
    assert isinstance(protein.molecule, oechem.OEGraphMol)
    protein = Protein.from_pdb("4yne", toolkit="MDAnalysis")
    assert isinstance(protein.molecule, Universe)


def test_lazy_protein():
    """Check lazy instantiation via PDB ID."""
    from kinoml.core.proteins import Protein

    protein = Protein(pdb_id="4yne")
    assert isinstance(protein._molecule, type(None))
    assert isinstance(protein.molecule, oechem.OEGraphMol)
    assert isinstance(protein._molecule, oechem.OEGraphMol)
    protein = Protein(pdb_id="4yne", toolkit="MDAnalysis")
    assert isinstance(protein.molecule, Universe)


def test_klifskinase_kinase_klifs_sequence():
    """Check access to kinase_klifs_sequence."""
    from kinoml.core.proteins import KLIFSKinase

    kinase = KLIFSKinase(uniprot_id="P04629")
    assert len(kinase.kinase_klifs_sequence) == 85
    assert isinstance(kinase.sequence, str)
    kinase = KLIFSKinase(kinase_klifs_id=480)
    assert len(kinase.kinase_klifs_sequence) == 85
    assert isinstance(kinase.sequence, str)
    kinase = KLIFSKinase(structure_klifs_id=3620)
    assert len(kinase.kinase_klifs_sequence) == 85
    assert isinstance(kinase.sequence, str)


def test_klifskinase_structure_klifs_sequence():
    """Check access to structure_klifs_sequence."""
    from kinoml.core.proteins import KLIFSKinase

    kinase = KLIFSKinase(structure_klifs_id=3620)
    assert len(kinase.structure_klifs_sequence) == 85


def test_klifskinase_structure_klifs_residues():
    """Check access to structure_klifs_residues."""
    from kinoml.core.proteins import KLIFSKinase

    kinase = KLIFSKinase(structure_klifs_id=3620)
    assert isinstance(kinase.structure_klifs_residues, pd.DataFrame) is True
    assert len(kinase.structure_klifs_residues) == 85
