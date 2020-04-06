from typing import Union, Iterable
from string import ascii_letters
import re
import logging
import os
from pathlib import Path

import requests

from .components import BaseProtein
from .sequences import Biosequence
from ..utils import FromDistpatcherMixin

logger = logging.getLogger(__name__)


class AminoAcidSequence(Biosequence, BaseProtein):
    """Biosequence that only allows proteinic aminoacids"""

    ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
    _ACCESSION_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=protein&id={}&rettype=fasta&retmode=text"


class ProteinStructure(BaseProtein):
    """
    Structural representation of a protein

    !!! todo
        This is probably going to be redone, so do not invest too much
    """

    def __init__(self, name=None):
        self.name = name

    @classmethod
    def from_file(cls, path, ext=None, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_sequence(cls, sequence, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_uniprot(cls, identifier, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_hgnc(cls, identifier, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_name(cls, identifier, **kwargs):
        raise NotImplementedError

    @property
    def sequence(self):
        s = "".join([r.symbol for r in self.residues])
        return AminoAcidSequence(s)


class Kinase(ProteinStructure):

    """
    Extends `Protein` to provide kinase-specific methods of
    instantiation.
    """

    @classmethod
    def from_klifs(cls, identifier, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_kinmap(cls, identifier, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_manning(cls, identifier, **kwargs):
        raise NotImplementedError
