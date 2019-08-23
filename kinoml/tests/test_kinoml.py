"""
Unit and regression test for the kinoml package.
"""

# Import package, test suite, and other packages as needed
import kinoml
import pytest
import sys

def test_kinoml_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "kinoml" in sys.modules
