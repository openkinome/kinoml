"""
Test kinoml.datasets.kinomescan
"""


def test_pkis2():
    from kinoml.datasets.pkis2 import PKIS2DatasetProvider

    provider = PKIS2DatasetProvider.from_source()
    assert len(provider.measurements) == 261_870
    assert (provider.measurements[0].values == 14.0).all()
    # check order in provider matches order in file
    assert (  # matches line 43 in file
        provider[17051].system.ligand.name
        == "O=C1NC(C2=C(C3=CC=CC=C3)C=C4C(C(C=C(O)C=C5)=C5N4)=C21)=O"
    )
    assert (  # matches line 44 in file
        provider[17052].system.ligand.name
        == "CN(N=C1)C=C1C(C=C2)=NN3C2=NN=C3[C@@H](C)C4=CC=C(N=CC=C5)C5=C4"
    )
