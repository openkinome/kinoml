"""
Test kinoml.core.components
"""


def test_components():
    from kinoml.core.components import MolecularComponent, BaseLigand, BaseProtein

    assert isinstance(BaseLigand(), MolecularComponent)
    assert isinstance(BaseProtein(), MolecularComponent)
