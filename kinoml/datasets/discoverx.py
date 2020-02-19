import logging

import pandas as pd

from .core import BaseDatasetProvider
from ..core.protein import AminoAcidSequence
from ..utils import grouper

logger = logging.getLogger(__name__)


class KINOMEScanDatasetProvider(BaseDatasetProvider):

    _RAW_DATASHEET = None

    @classmethod
    def from_source(cls, filename=None, **kwargs):
        if filename is None:
            filename = cls._RAW_DATASHEET
        if filename is None:
            raise NotImplementedError(
                "No default datasheet has been specified in this class."
            )
        df = pd.read_csv(filename)

        # Goal 1: obtain chemical_data (list of MolecularSystem)

        # Goal 2: obtain clean measurements

    @staticmethod
    def _retrieve_sequence(*accessions):
        # 1) Retrieve all raw sequences from NCBI Protein db
        sequences = []
        for accessions in grouper(
            accessions, AminoAcidSequence.ACCESSION_MAX_RETRIEVAL, fillvalue=""
        ):
            sequences.extend(AminoAcidSequence.from_accession(*accessions))
        return sequences

