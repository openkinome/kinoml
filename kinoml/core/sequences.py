"""
Sequence-like objects to build MolecularComponents and others.
"""
import logging
import re
from string import ascii_letters


logger = logging.getLogger(__name__)


class Biosequence(object):
    """
    Base class for string representations of biological polymers
    (nucleic acids, peptides, proteins...).

    Parameters
    ----------
    sequence: str, default=""
        The sequence in one-letter codes.
    name: str, default=""
        The sequence name.
    metadata: dict or None, default=None
        Additional data as a dictionary.
    """

    ALPHABET = set(ascii_letters)

    def __init__(self, sequence="", name="", metadata=None, **kwargs):
        diff = set(sequence).difference(self.ALPHABET)
        if diff:
            raise ValueError(
                f"Biosequence can only contain characters in {self.ALPHABET}, "
                f"but found these extra ones: {diff}."
            )
        self._sequence = sequence
        self.name = name
        self.metadata = {"sequence_source": "user"}
        if metadata is not None:
            self.metadata.update(metadata)

    @property
    def sequence(self):
        return self._sequence

    @sequence.setter
    def sequence(self, new_value):
        self._sequence = new_value

    @sequence.getter
    def sequence(self):
        if len(self._sequence) == 0:
            self._query_sequence_sources()
        return self._sequence

    def _query_sequence_sources(self):
        """
        Query available sources for sequence details. Overwrite method in subclasses to fetch
        data.
        """
        pass

    def substitute(self, substitution):
        """
        Given ``XYYYZ``, substitute element ``X`` at position ``YYY`` with ``Z``, e.g. C1156Y.

        Parameters
        ----------
        substitution: str
            Substitution to apply. It must be formatted as
            ``[existing element][1-indexed position][new element]``.

        Examples
        --------
        >>> s = Biosequence(sequence="ABCD")
        >>> s.sequence
        "ABCD"
        >>> s.substitute("B2F")
        >>> s.sequence
        "AFCD"
        """
        search = re.search(r"([A-Z])(\d+)([A-Z])", substitution)
        if search is None:
            raise ValueError(f"Mutation `{substitution}` is not a valid substitution.")
        old, position, new = search.groups()
        position = int(position)
        assert (
            new in self.ALPHABET
        ), f"{new} is not a valid {self.__class__.__name__} character ({self.ALPHABET})"
        if position < 1 or position > len(self.sequence):
            raise ValueError(
                f"Cannot find position {position} in the sequence "
                f"for substitution `{substitution}`."
            )
        if self.sequence[position - 1] != old:
            raise ValueError(
                f"Cannot find {old} at position {position} for substitution `{substitution}`,"
                f" found {self.sequence[position - 1]} instead."
            )
        self.sequence = f"{self.sequence[: position - 1]}{new}{self.sequence[position:]}"
        if "mutations" in self.metadata.keys():
            self.metadata["mutations"] += f" sub{substitution}"
        else:
            self.metadata["mutations"] = f"sub{substitution}"

    def delete(self, first, last, insert=""):
        """
        Delete all elements between first and last positions including bounds. Optionally, provide
        an additional insert that shell be placed at the position of the deletion.

        Parameters
        ----------
        first: int
            First residue to delete (1-indexed).
        last: int
            Last residue to delete (1-indexed).
        insert: str, default=""
            Sequence that should be placed at the position of the deletion.

        Examples
        --------
        >>> s = Biosequence(sequence="ABCD")
        >>> s.sequence
        "ABCD"
        >>> s.delete(3,3, insert="GH")
        >>> s.sequence
        "ABGHD"
        """
        assert all(new in self.ALPHABET for new in insert)
        if first < 1 or last > len(self.sequence):
            raise ValueError(f"Deletion {first}-{last} out of bounds for given sequence.")
        self.sequence = f"{self.sequence[: first - 1]}{insert}{self.sequence[last:]}"
        if "mutations" in self.metadata.keys():
            self.metadata["mutations"] += f" del{first}-{last}{insert}"
        else:
            self.metadata["mutations"] = f"del{first}-{last}{insert}"

    def insert(self, position, insert):
        """
        Insert a sequence at the given position.

        Parameters
        ----------
        position: int
            Position (1-indexed) to place the insertion.
        insert: str
            The sequence of the insertion.

        Examples
        --------
        >>> s = Biosequence(sequence="ABCD")
        >>> s.sequence
        "ABCD"
        >>> s.insert(4, insert="EF")
        >>> s.sequence
        "ABCEFD"
        """
        assert all(new in self.ALPHABET for new in insert)
        if position < 1 or position - 1 > len(self.sequence):
            raise ValueError(f"Insertion position {position} out of bonds for given sequence.")
        self.sequence = f"{self.sequence[: position - 1]}{insert}{self.sequence[position:]}"
        if "mutations" in self.metadata.keys():
            self.metadata["mutations"] += f" ins{position}{insert}"
        else:
            self.metadata["mutations"] = f"ins{position}{insert}"


class AminoAcidSequence(Biosequence):
    """
    Biosequence for amino acid sequences.

    Parameters
    ----------
    uniprot_id: str or None, default=None
        The UniProt ID.
    ncbi_id: str or None, default=None
        The NCBI ID.
    sequence: str, default=""
        The amino acid sequence in one-letter codes.
    name: str, default=""
        The sequence name.
    metadata: dict or None, default=None
        Additional data as a dictionary.

    Examples
    --------
    Amino acid sequences can be created by providing the sequence manually or by fetching from
    e.g. UniProt:

    >>> alatripeptide = AminoAcidSequence(sequence="AAA", name="alatripeptide")
    >>> alatripeptide.sequence
    "AAA"
    >>> abl1 = AminoAcidSequence(uniprot_id="P00519", name="ABL1")
    >>> abl1.sequence[:5]
    "MLEIC"

    Fetched sequences can be altered by providing information via metadata["mutations"], i.e.:
     - insertions - formatted like "ins123AGA"
     - deletions - formatted like "del12-15P" (the P stands for a proline insert (optional))
     - substitutions - formatted like "T315A"

     >>> abl1 = AminoAcidSequence(
     >>>     uniprot_id="P00519", name="ABL1", metadata={"mutations": "T315A"}
     >>> )

     Multiple mutations can be added sequentially in a single string:

     >>> abl1 = AminoAcidSequence(
     >>>     uniprot_id="P00519", name="ABL1", metadata={"mutations": "T315A del320-22P"}
     >>> )

     An artificial contruct only consisting of a part of the sequence can be specified via
     metadata["construct_range"]:
     >>> abl1 = AminoAcidSequence(
     >>>     uniprot_id="P00519",
     >>>     name="ABL1",
     >>>     metadata={"mutations": "T315A", "construct_range": "229-512"}
     >>> )

    """

    ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

    def __init__(self, uniprot_id="", ncbi_id="", sequence="", name="", metadata=None, **kwargs):
        super().__init__(sequence=sequence, name=name, metadata=metadata, **kwargs)
        self.uniprot_id = uniprot_id
        self.ncbi_id = ncbi_id

    def _query_sequence_sources(self):
        """
        Query available sources for sequence details. Add additional methods below to allow
        fetching from other sources. Perform mutations etc if given via metadata.
        """
        if self.uniprot_id:
            self._query_uniprot()
        elif self.ncbi_id:
            self._query_ncbi()
        if "mutations" in self.metadata.keys():
            mutations = self.metadata["mutations"].split()
            del self.metadata["mutations"]  # remove mutations, will be added subsequently
            for mutation in mutations:
                import re

                if mutation.startswith("ins"):  # insertion
                    logger.debug(f"Performing insertion {mutation} ...")
                    match = re.search("ins(?P<position>[0-9]+)(?P<insertion>[A-Z]+)", mutation)
                    self.insert(int(match.group("position")), match.group("insertion"))
                elif mutation.startswith("del"):  # deletion
                    logger.debug(f"Performing deletion {mutation} ...")
                    match = re.search(
                        "del(?P<first>[0-9]+)-(?P<last>[0-9]+)(?P<insertion>[A-Z]*)",
                        mutation,
                    )
                    self.delete(
                        int(match.group("first")),
                        int(match.group("last")),
                        match.group("insertion"),
                    )
                else:  # substitution
                    logger.debug(f"Performing substitution {mutation} ...")
                    self.substitute(mutation)
        if "construct_range" in self.metadata.keys():
            logger.debug(f"Cropping sequence to construct {self.metadata['construct_range']} ...")
            first, last = [int(x) for x in self.metadata["construct_range"].split("-")]
            self._sequence = self._sequence[first - 1 : last]  # 1-indexed

    def _query_uniprot(self):
        """Fetch the amino acid sequence from UniProt."""
        import requests
        import json

        response = requests.get(f"https://www.ebi.ac.uk/proteins/api/proteins/{self.uniprot_id}")
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch sequence for UniProt ID {self.uniprot_id}")

        response = json.loads(response.text)
        self._sequence = response["sequence"]["sequence"]
        self.metadata["sequence_source"] = "UniProt"

    def _query_ncbi(self):
        """Fetch the amino acid sequence from NCBI."""
        import requests

        response = requests.get(
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
            f"db=protein&id={self.ncbi_id}&rettype=fasta&retmode=text"
        )
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch sequence for NCBI ID {self.ncbi_id}")

        self._sequence = "".join(response.text.split("\n")[1:])
        self.metadata["sequence_source"] = "NCBI"

    @staticmethod
    def ncbi_to_uniprot(ncbi_id):
        """
        Convert an NCBI protein accession to the corresponding UniProt ID.

        Parameters
        ----------
        ncbi_id: str
            The NCBI protein accession.

        Returns
        -------
        : str
            The corresponding UniProt ID, empty string if not successful.
        """

        import requests

        url = "https://www.uniprot.org/uploadlists/"
        params = {"from": "P_REFSEQ_AC", "to": "SWISSPROT", "format": "tab", "query": ncbi_id}
        response = requests.get(url, params=params)
        response = response.text.split("\n")
        if len(response) != 3:
            return ""
        return response[1].split("\t")[1]


class DNASequence(Biosequence):
    """Biosequence that only allows DNA bases."""

    ALPHABET = "ATCG"


class RNASequence(Biosequence):
    """Biosequence that only allows RNA bases."""

    ALPHABET = "AUCG"
