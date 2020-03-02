"""
KinoML
Machine Learning for kinase modeling
"""

# Handle versioneer
from ._version import get_versions

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions


def function(arg1, kwarg=None):
    """
    Example function

    Parameters:
        arg1:
            Yeah
        kwarg:
            None

    Returns:
        smth:
            Something
    """
    pass
