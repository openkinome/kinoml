"""
Test complex featurizers of `kinoml.features`
"""
import pandas as pd


def test_oecomplexfeaturizer():
    """Check OEComplexFeaturizer with different inputs."""
    from kinoml.core.ligands import Ligand
    from kinoml.core.proteins import Protein
    from kinoml.core.systems import ProteinLigandComplex
    from kinoml.features.complexes import OEComplexFeaturizer

    systems = []
    protein = Protein(pdb_id="4f8o", name="PsaA")
    ligand = Ligand(name="AEBSF")
    system = ProteinLigandComplex(components=[protein, ligand])
    systems.append(system)
    protein = Protein.from_pdb(pdb_id="4f8o", name="PsaA")
    protein.uniprot_id = "P31522"
    protein.chain_id = "A"
    protein.alternate_location = "B"
    protein.expo_id = "AES"
    ligand = Ligand(name="AEBSF")
    system = ProteinLigandComplex(components=[protein, ligand])
    systems.append(system)
    featurizer = OEComplexFeaturizer()
    systems = featurizer.featurize(systems)
    # check LIG exists
    assert len(systems[0].featurizations["last"].select_atoms("resname LIG").residues) == 1
    assert len(systems[1].featurizations["last"].select_atoms("resname LIG").residues) == 1
    # check caps
    assert (
        len(systems[0].featurizations["last"].select_atoms("resname ACE or resname NME").residues)
        == 2
    )
    assert (
        len(systems[1].featurizations["last"].select_atoms("resname ACE or resname NME").residues)
        == 1
    )
    # check number of residues
    assert len(systems[0].featurizations["last"].residues) == 240
    assert len(systems[1].featurizations["last"].residues) == 217
    # check numbering of first residue
    assert systems[0].featurizations["last"].residues[0].resid == 1
    assert systems[1].featurizations["last"].residues[0].resid == 44


def test_oedockingfeaturizer_fred():
    """Check OEDockingFeaturizer with Fred and different inputs."""
    from kinoml.core.ligands import Ligand
    from kinoml.core.proteins import Protein
    from kinoml.core.systems import ProteinLigandComplex
    from kinoml.features.complexes import OEDockingFeaturizer

    systems = []
    # define the binding site for docking via co-crystallized ligand
    protein = Protein(pdb_id="4yne", name="NTRK1")
    protein.expo_id = "4EK"
    ligand = Ligand(
        smiles="C1CC(N(C1)C2=NC3=C(C=NN3C=C2)NC(=O)N4CCC(C4)O)C5=C(C=CC(=C5)F)F",
        name="larotrectinib",
    )
    system = ProteinLigandComplex(components=[protein, ligand])
    systems.append(system)
    # define the binding site for docking via residue IDs
    protein = Protein(pdb_id="4yne", name="NTRK1")
    protein.pocket_resids = [
        516,
        517,
        521,
        524,
        542,
        544,
        573,
        589,
        590,
        591,
        592,
        595,
        596,
        654,
        655,
        656,
        657,
        667,
        668,
    ]
    ligand = Ligand(
        smiles="C1CC(N(C1)C2=NC3=C(C=NN3C=C2)NC(=O)N4CCC(C4)O)C5=C(C=CC(=C5)F)F",
        name="larotrectinib_2",
    )
    system = ProteinLigandComplex(components=[protein, ligand])
    systems.append(system)
    featurizer = OEDockingFeaturizer(method="Fred")
    systems = featurizer.featurize(systems)
    # check docking score was stored
    assert isinstance(systems[0].featurizations["last"]._topology.docking_score, float)
    # check LIG exists
    assert len(systems[0].featurizations["last"].select_atoms("resname LIG").residues) == 1
    assert len(systems[1].featurizations["last"].select_atoms("resname LIG").residues) == 1
    # check caps
    assert (
        len(systems[0].featurizations["last"].select_atoms("resname ACE or resname NME").residues)
        == 10
    )
    assert (
        len(systems[1].featurizations["last"].select_atoms("resname ACE or resname NME").residues)
        == 10
    )
    # check numbering of first residue
    assert systems[0].featurizations["last"].residues[0].resid == 501
    assert systems[1].featurizations["last"].residues[0].resid == 501


def test_oedockingfeaturizer_hybrid():
    """Check OEDockingFeaturizer with Hybrid."""
    from kinoml.core.ligands import Ligand
    from kinoml.core.proteins import Protein
    from kinoml.core.systems import ProteinLigandComplex
    from kinoml.features.complexes import OEDockingFeaturizer

    systems = []
    protein = Protein(pdb_id="4yne", name="NTRK1")
    protein.expo_id = "4EK"
    ligand = Ligand(
        smiles="C1CC(N(C1)C2=NC3=C(C=NN3C=C2)NC(=O)N4CCC(C4)O)C5=C(C=CC(=C5)F)F",
        name="larotrectinib",
    )
    system = ProteinLigandComplex(components=[protein, ligand])
    systems.append(system)
    featurizer = OEDockingFeaturizer(method="Hybrid")
    systems = featurizer.featurize(systems)
    # check LIG exists
    assert len(systems[0].featurizations["last"].select_atoms("resname LIG").residues) == 1
    # check caps
    assert (
        len(systems[0].featurizations["last"].select_atoms("resname ACE or resname NME").residues)
        == 10
    )
    # check numbering of first residue
    assert systems[0].featurizations["last"].residues[0].resid == 501


def test_oedockingfeaturizer_posit():
    """Check OEDockingFeaturizer with Posit."""
    from kinoml.core.ligands import Ligand
    from kinoml.core.proteins import Protein
    from kinoml.core.systems import ProteinLigandComplex
    from kinoml.features.complexes import OEDockingFeaturizer

    systems = []
    protein = Protein(pdb_id="4yne", name="NTRK1")
    protein.expo_id = "4EK"
    ligand = Ligand(
        smiles="C1CC(N(C1)C2=NC3=C(C=NN3C=C2)NC(=O)N4CCC(C4)O)C5=C(C=CC(=C5)F)F",
        name="larotrectinib",
    )
    system = ProteinLigandComplex(components=[protein, ligand])
    systems.append(system)
    featurizer = OEDockingFeaturizer(method="Posit")
    systems = featurizer.featurize(systems)
    # check LIG exists
    assert len(systems[0].featurizations["last"].select_atoms("resname LIG").residues) == 1
    # check caps
    assert (
        len(systems[0].featurizations["last"].select_atoms("resname ACE or resname NME").residues)
        == 10
    )
    # check numbering of first residue
    assert systems[0].featurizations["last"].residues[0].resid == 501
    # check posit probability was stored
    assert isinstance(systems[0].featurizations["last"]._topology.posit_probability, float)


def test_mostsimilarpdbligandfeaturizer():
    """Check MostSimilarPDBLigandFeaturizer with different similarity metrics."""
    from kinoml.core.ligands import Ligand
    from kinoml.core.proteins import Protein
    from kinoml.core.systems import ProteinLigandComplex
    from kinoml.features.complexes import MostSimilarPDBLigandFeaturizer

    for metric in ["mcs", "fingerprint", "openeye_shape"]:
        systems = []
        protein = Protein(uniprot_id="P04629", name="NTRK1")
        ligand = Ligand(
            smiles="C1CC(N(C1)C2=NC3=C(C=NN3C=C2)NC(=O)N4CCC(C4)O)C5=C(C=CC(=C5)F)F",
            name="larotrectinib",
        )
        system = ProteinLigandComplex(components=[protein, ligand])
        systems.append(system)
        featurizer = MostSimilarPDBLigandFeaturizer(similarity_metric=metric)
        systems = featurizer.featurize(systems)
        assert isinstance(systems[0].protein.pdb_id, str)
        assert isinstance(systems[0].protein.chain_id, str)
        assert isinstance(systems[0].protein.expo_id, str)


def test_klifsconformationtemplatesfeaturizer():
    """Check KLIFSConformationTemplatesFeaturizer with fingerprint only."""
    from kinoml.core.ligands import Ligand
    from kinoml.core.proteins import KLIFSKinase
    from kinoml.core.systems import ProteinLigandComplex
    from kinoml.features.complexes import KLIFSConformationTemplatesFeaturizer

    systems = []
    protein = KLIFSKinase(uniprot_id="P04629", name="NTRK1")
    ligand = Ligand(
        smiles="C1CC(N(C1)C2=NC3=C(C=NN3C=C2)NC(=O)N4CCC(C4)O)C5=C(C=CC(=C5)F)F",
        name="larotrectinib",
    )
    system = ProteinLigandComplex(components=[protein, ligand])
    systems.append(system)
    featurizer = KLIFSConformationTemplatesFeaturizer(similarity_metric="fingerprint")
    systems = featurizer.featurize(systems)
    # check feature is dataframe
    assert isinstance(systems[0].featurizations["last"], pd.DataFrame)
    # check dataframe is not empty
    assert len(systems[0].featurizations["last"]) > 0
