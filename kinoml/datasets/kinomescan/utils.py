import re
import logging
from pathlib import Path
import pandas as pd

from ...core.protein import AminoAcidSequence
from ...utils import grouper, datapath, APPDIR

logger = logging.getLogger(__name__)


class KINOMEScanMapper:

    _SEQUENCE_INFORMATION = datapath(
        "kinomescan/DiscoverX_489_Kinase_Assay_Construct_Information.csv"
    )

    def __init__(self, use_cache=True, **kwargs):
        cached_path = (
            Path(APPDIR.user_cache_dir) / "kinomescan" / "kinomescanmapper.csv"
        )
        if use_cache and cached_path.is_file():
            df = pd.read_csv(cached_path, index_col=0)
            assert (df.columns == ["accession", "raw_sequence", "sequence"]).all()
            self._data = df
        else:
            raw_df = pd.read_csv(self._SEQUENCE_INFORMATION)
            accessions = raw_df["Accession Number"].tolist()
            mutations = raw_df["Construct Description"].tolist()
            names = raw_df["DiscoverX Gene Symbol"].tolist()
            start_stop_strings = raw_df["AA Start/Stop"].tolist()
            wt_kinases, kinases = self._obtain_sequences(
                accessions, mutations, start_stop_strings
            )
            rows = [["name", "accession", "raw_sequence", "sequence"]]
            rows.extend(zip(names, accessions, wt_kinases, kinases))

            self._data = pd.DataFrame.from_records(
                data=rows[1:], columns=rows[0], index="name"
            )
            cached_path.parent.mkdir(parents=True, exist_ok=True)
            self._data.to_csv(cached_path)

    def sequence_for_name(self, name):
        return self._data.loc[name, "sequence"]

    def sequence_for_accession(self, accession):
        return self._data[self._data.accession == accession].sequence

    def _obtain_sequences(self, accessions, mutations, start_stop_strings):
        # 1) Retrieve all raw kinase sequences from NCBI Protein db
        wt_sequences = self._retrieve_sequence(*accessions)
        # 2) Apply mutations to wild-type sequences & cut
        kinases = []
        for seq, mut, start_stop in zip(wt_sequences, mutations, start_stop_strings):
            seq = self._apply_mutations(seq, mut)
            kinase = self._cut_sequence(seq, start_stop)
            # in this dataset, a kinase is represented by just its sequence
            kinases.append(kinase)

        return wt_sequences, kinases

    @staticmethod
    def _retrieve_sequence(*accessions):
        # 1) Retrieve all raw sequences from NCBI Protein db
        sequences = []
        max_requests = AminoAcidSequence.ACCESSION_MAX_RETRIEVAL
        for accession in grouper(accessions, max_requests, fillvalue=""):
            sequences.extend(AminoAcidSequence.from_accession(*accession))
        return sequences

    @staticmethod
    def _apply_mutations(sequence, mutation_string):
        if mutation_string is None:
            return sequence
        search = re.search(r"Mutation\s?\((.*)\)", mutation_string)
        if search is None:
            return sequence
        mutations = search.group(1).split(",")
        return sequence.mutate(*mutations, raise_errors=False)

    @staticmethod
    def _cut_sequence(sequence, start_stop_string):
        if start_stop_string in (None, "Null", "null"):
            return sequence
        start, stop = start_stop_string.split("/")
        return sequence.cut(start, stop)
