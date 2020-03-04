"""
KinoML is a structure-informed machine learning library
with a focus on kinase modeling
"""

# Handle versioneer
from ._version import get_versions

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions

