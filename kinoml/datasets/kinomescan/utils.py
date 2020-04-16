import re
import logging
from pathlib import Path
import pandas as pd
from typing import Union, AnyStr, Iterable
from time import sleep

from ...core.proteins import AminoAcidSequence
from ...utils import grouper, datapath, APPDIR

logger = logging.getLogger(__name__)


class KINOMEScanMapper:

    """
    This helper class helps retrieve sequence information out of the raw
    data provided by DiscoverX, which only offers NCBI accessions, mutations
    and construct limits. We process this to obtain a FASTA sequence that
    can be ingested in our pipelines.

    Since this relies on online queries, it will cache the results to disk
    by default.

    Parameters:
        raw_datasheet: Raw CSV file with the DiscoverX information
        use_cache: Whether to read the data from cache if possible. Set to
            `False` to ignore existing caches and rewrite them.

    """

    _version = "202004161800"  # update this to current datetime if this class changes
    _columns = "name", "accession", "raw_sequence", "mutations", "start_stop", "sequence"

    def __init__(
        self,
        raw_datasheet: Union[AnyStr, Path] = datapath(
            "kinomescan/DiscoverX_489_Kinase_Assay_Construct_Information.csv"
        ),
        use_cache: bool = True,
        **kwargs,
    ):
        cached_path = Path(APPDIR.user_cache_dir) / "kinomescan" / f"mapper.{self._version}.csv"
        self.sequence_information = raw_datasheet
        if use_cache and cached_path.is_file():
            df = pd.read_csv(cached_path, index_col=0)
            assert (df.columns == list(self._columns[1:])).all()
            self._data = df
        else:
            raw_df = pd.read_csv(self.sequence_information)
            accessions = raw_df["Accession Number"].tolist()
            mutation_strings = raw_df["Construct Description"].tolist()
            names = raw_df["DiscoverX Gene Symbol"].tolist()
            start_stop_strings = raw_df["AA Start/Stop"].tolist()
            wt_kinases, kinases, mutations, start_stops = self._obtain_sequences(
                accessions, mutation_strings, start_stop_strings
            )
            rows = list(zip(names, accessions, wt_kinases, mutations, start_stops, kinases))
            self._data = pd.DataFrame.from_records(data=rows, columns=self._columns, index="name")
            cached_path.parent.mkdir(parents=True, exist_ok=True)
            self._data.to_csv(cached_path)

    def name_is_mutated(self, name: AnyStr):
        return self._data.loc[name, "mutation"] in (None, pd.np.nan, float("nan"), "NaN")

    def sequence_for_name(self, name: AnyStr):
        """
        Given a kinase name, return the corresponding FASTA sequence
        """
        return self._data.loc[name, "sequence"]

    def accession_for_name(self, name: AnyStr):
        """
        Given a kinase name, return the corresponding NCBI accession
        """
        return self._data.loc[name, "accession"]

    def mutations_for_name(self, name: AnyStr):
        """
        Given a kinase name, return the corresponding mutations
        """
        return self._data.loc[name, "mutations"]

    def start_stop_for_name(self, name: AnyStr):
        """
        Given a kinase name, return the corresponding start&stop positions
        """
        return self._data.loc[name, "start_stop"]

    def sequence_for_accession(self, accession: AnyStr):
        """
        Given a NCBI identifier, return the corresponding FASTA sequence
        """
        return self._data[self._data.accession == accession].sequence.values

    def _obtain_sequences(
        self,
        accessions: Iterable[AnyStr],
        mutation_strings: Iterable[AnyStr],
        start_stop_strings: Iterable[AnyStr],
    ) -> [[AminoAcidSequence], [AminoAcidSequence], [AnyStr], [AnyStr]]:
        """
        Main method to retrieve processed sequences from the datasheet.
        Three steps:
        1) Retrieve all raw kinase sequences from NCBI Protein db
        2) Apply mutations to wild-type sequences
        3) Cut to specified length

        Parameters:
            accessions: list of NCBI identifiers
            mutation_strings: String containing WT/mutated status of that entry. This
                needs to be parsed before delegating to `Biosequence`
            start_stop_strings: String containing the subsequence used in the
                assay. This needs to be parsed before delegating to `Biosequence`

        Returns:
            wild type (unprocessed) sequences, processed sequences
        """
        wt_sequences = self._retrieve_sequence(*accessions)
        kinases, mutations, start_stops = [], [], []
        for seq, mut, start_stop in zip(wt_sequences, mutation_strings, start_stop_strings):
            seq, mut = self._apply_mutations(seq, mut)
            kinase, start_stop = self._cut_sequence(seq, start_stop)
            kinases.append(kinase)
            mutations.append(mut)
            start_stops.append(start_stop)

        return wt_sequences, kinases, mutations, start_stops

    @staticmethod
    def _retrieve_sequence(*accessions: Iterable[AnyStr]):
        """
        Batch all queries in groups and delegate to
        `AminoAcidSequence.from_ncbi`.
        """
        # 1) Retrieve all raw sequences from NCBI Protein db
        sequences = []
        max_requests = AminoAcidSequence.ACCESSION_MAX_RETRIEVAL
        for accession in grouper(accessions, max_requests, fillvalue=""):
            sequences.extend(AminoAcidSequence.from_ncbi(*accession))
            sleep(0.5)
        return sequences

    @staticmethod
    def _apply_mutations(sequence: AminoAcidSequence, mutation_string: AnyStr):
        """
        Parse mutation strings and delegate to `AminoAcidSequence.mutate()`.

        Expected format is `Mutation (A123B,C456D)`.
        """
        if mutation_string is None:
            return sequence, None
        search = re.search(r"Mutation\s?\((.*)\)", mutation_string)
        if search is None:
            return sequence, None
        mutations = search.group(1).split(",")
        return sequence.mutate(*mutations, raise_errors=False), mutations

    @staticmethod
    def _cut_sequence(sequence: AminoAcidSequence, start_stop_string):
        if start_stop_string in (None, "Null", "null"):
            return sequence, None
        start, stop = start_stop_string.split("/")
        return sequence.cut(start, stop), (start, stop)
