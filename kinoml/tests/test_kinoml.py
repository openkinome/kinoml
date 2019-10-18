"""
Unit and regression test for the kinoml package.
"""

import kinoml
import sys


def test_kinoml_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "kinoml" in sys.modules
