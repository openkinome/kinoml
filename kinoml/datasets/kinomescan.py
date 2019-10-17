"""
Handlers for datasets based on KINOMEscan assays.
"""
import re
from collections import Counter
from operator import itemgetter

import pandas as pd
from kinoml.datasets.utils import Biosequence, AminoAcidSequence
from kinoml.utils import datapath, grouper


class KINOMEscan:

    """
    Process the DiscoverX spreadsheet to programmatically:
        - obtain sequences from NCBI Protein with accession
        - apply specified mutations in 'Construct Description'
        - cut full sequence to construct as specified in 'AA Start/Stop'

    TODO: Check if assumptions done at ``kinoml.datasets.utils.Biosequence`` are correct.

    """

    _raw_spreadsheet = datapath("kinomescan/DiscoverX_489_Kinase_Assay_Construct_Information.csv")

    def __init__(self):
        self.data = pd.read_csv(self._raw_spreadsheet)

    def retrieve_sequences(self, mutations=True, cut=True):
        """
        Process accessions to obtain sequence, apply mutations and cut to final length.

        Modifies ``self.data`` in-place.
        """
        # 1) Retrieve all raw sequences from NCBI Protein db
        all_accessions = self.data['Accession Number'].tolist()
        all_sequences = []
        for accessions in grouper(all_accessions, AminoAcidSequence.ACCESSION_MAX_RETRIEVAL, fillvalue=''):
            all_sequences.extend(AminoAcidSequence.from_accession(*accessions))
        self.data['full_sequence'] = list(map(str, all_sequences))

        if not mutations:
            return
        # 2) Apply mutations
        # We are using AminoAcidSequence.mutate(...) for that
        mutated_sequences = []
        mutation_strings = self.data['Construct Description'].tolist()
        for sequence, mutation_string in zip(all_sequences, mutation_strings):
            search = re.search(r'Mutation\s?\((.*)\)', mutation_string)
            if search is None:
                mutated_sequences.append(sequence)
                continue
            mutations = search.group(1).split(',')
            mutated_sequences.append(sequence.mutate(*mutations, raise_errors=False))
        self.data['mutated_sequence'] = list(map(str, mutated_sequences))

        if not cut:
            return
        # 3) Cut sequence
        cut_sequences = []
        start_stop_strings = self.data['AA Start/Stop'].tolist()
        for sequence, start_stop_string in zip(mutated_sequences, start_stop_strings):
            if start_stop_string in (None, "Null", "null") or sequence is None:
                cut_sequences.append(sequence)
                continue
            start, stop = start_stop_string.split("/")
            cut_sequences.append(sequence.cut(start, stop))
        self.data['cut_sequence'] = list(map(str, cut_sequences))



if __name__ == "__main__":
    dataset = KINOMEscan()
    dataset.retrieve_sequences()
    dataset.data.to_csv(datapath("kinomescan/DiscoverX_489_Kinase_Assay_Construct_Information+sequences.csv"))
