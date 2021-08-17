"""
Unit and regression test for the kinoml package.
"""

# Import package, test suite, and other packages as needed
import kinoml  # pylint: disable=unused-import
import sys


def test_kinoml_imported():
    """
    Sample test, will always pass so long as import statement worked
    """
    assert "kinoml" in sys.modules


def test_3rdparty_imports():
    """
    Some packages can be tricky to install. Make sure we can import them.
    """
    import torch  # pylint: disable=unused-import

    assert "torch" in sys.modules

    import torch_geometric  # pylint: disable=unused-import

    assert "torch_geometric" in sys.modules
