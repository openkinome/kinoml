"""
Sequence-like objects to build MolecularComponents and others.
"""
from string import ascii_letters
from collections import Counter
import logging
import re
import json
from typing import Union, Iterable

import requests

logger = logging.getLogger(__name__)


class Biosequence(str):
    """
    Base class for string representations of biological polymers
    (nucleic acids, peptides, proteins...)

    Note
    ----
    How to handle several mutations at the same time, while
    keeping indices relevant (after a deletion, a replacement
    or insertion position might be wrong).
    """

    ALPHABET = set(ascii_letters)
    _ACCESSION_URL = None
    ACCESSION_MAX_RETRIEVAL = 50

    def __new__(cls, value, name="", metadata=None, *args, **kwargs):
        """
        We are subclassing ``str`` to:

        - provide a ``.metadata`` dict
        - validate input is part of the allowed alphabet
        """
        diff = set(value).difference(cls.ALPHABET)
        if diff:
            raise ValueError(
                f"Biosequence can only contain characters in {cls.ALPHABET}, "
                f"but found these extra ones: {diff}."
            )
        s = super().__new__(cls, value, *args, **kwargs)
        s.name = name
        s.sequence = value
        s.metadata = {}
        # TODO: We might override some metadata data with this blind update
        if metadata is not None:
            s.metadata.update(metadata)
        return s

    @classmethod
    def from_ncbi(
        cls,
        *accessions: str,
    ) -> Union["Biosequence", Iterable["Biosequence"]]:
        """
        Get FASTA sequence from an online NCBI identifier

        Parameters
        ----------
        accessions : str
            NCBI identifier. Multiple can be provided!

        Returns
        -------
        Retrieved biosequence(s)

        Examples
        --------
        >>> sequence = AminoAcidSequence.from_ncbi("AAC05299.1")
        >>> print(sequence[:10])
        MSVNSEKSSS
        >>> print(sequence.name)
        AAC05299.1 serine kinase SRPK2 [Homo sapiens]
        """
        if cls._ACCESSION_URL is None:
            raise NotImplementedError
        if len(accessions) > cls.ACCESSION_MAX_RETRIEVAL:
            raise ValueError(
                f"You can only provide {cls.ACCESSION_MAX_RETRIEVAL} accessions at the same time."
            )
        r = requests.get(cls._ACCESSION_URL.format(",".join(accessions)))
        r.raise_for_status()
        sequences = []
        for line in r.text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                sequences.append({"name": line[1:], "sequence": []})
            else:
                sequences[-1]["sequence"].append(line)
        if not sequences:
            return
        objects = []
        for sequence, accession in zip(sequences, accessions):
            obj = cls(
                "".join(sequence["sequence"]),
                name=sequence["name"],
                metadata={"accession": accession},
            )
            objects.append(obj)
        if not objects:
            return None
        if len(objects) == 1:
            return objects[0]
        return objects

    def cut(self, start: str, stop: str, check: bool = True) -> "Biosequence":
        """
        Slice a sequence using biological notation

        Parameters
        ----------
        start : str
            Starting element and 1-indexed position; e.g. C123
        stop : str
            Ending element and 1-indexed position; e.g. T234
            This will be included in the resulting sequence
        check : bool, optional=True
            Whether to test if the existing elements correspond
            to those specified in the bounds

        Returns
        -------
        Biosequence
            Substring corresponding to [start, end]. Right bound is included!

        Examples
        --------
        >>> s = Biosequence("ATCGTHCTCH")
        >>> s.cut("T2", "T8")
        "TCGTHCT"
        """
        start_res, start_pos = start[0], int(start[1:])
        stop_res, stop_pos = stop[0], int(stop[1:])
        if check:
            assert (
                start_res == self[start_pos - 1]
            ), f"Element at position {start_pos} is not {start_res}"
            assert (
                stop_res == self[stop_pos - 1]
            ), f"Element at position {stop_pos} is not {stop_res}"
        return self.__class__(
            self[start_pos - 1 : stop_pos],
            name=f"{self.name}{ ' | ' if self.name else '' }Cut: {start}/{stop}",
            metadata={"cut": (start, stop)},
        )

    def mutate(self, *mutations: str, raise_errors: bool = True) -> "Biosequence":
        """
        Apply a mutation on the sequence using biological notation.

        Parameters
        ----------
        mutations : str
            Mutations to be applied. Indices are always 1-indexed. It can be one of:
            (1) substitution, like ``C234T`` (C at position 234 will be replaced by T);
            (2) deletion, like ``L746-A750del`` (delete everything between L at position 746
            A at position 750, bounds not included);
            (3) insertion, like ``1151Tins`` (insert a T after position 1151)
        raise_errors : bool, optional=True
            Raise ``ValueError`` if one of the mutations is not supported.

        Returns
        -------
        Biosequence
            The edited sequence

        Examples
        --------
        >>> s = Biosequence("ATCGTHCTCH")
        >>> s.mutate("C3P")
        "ATPGTHCTCH"
        >>> s.mutate("T2-T5del")
        "ATTHCTCH"
        >>> s.mutate("5Tins")
        "ATCGTTHCTCH"
        """
        # We can only handle one insertion or deletion at once now
        mutation_types = {m: self._type_mutation(m, raise_errors) for m in mutations}
        mutation_count = Counter(mutation_types.values())
        if mutation_count["insertion"] + mutation_count["deletion"] > 1:
            msg = f"Only one simultaneous insertion or deletion is currently supported. You provided `{','.join(mutations)}`"
            if raise_errors:
                raise ValueError(msg)
            logger.warning("Warning: %s", msg)
            return None

        # Reverse alphabetical order (substitutions will come first)
        mutated = self
        for mutation in sorted(mutations, key=lambda m: mutation_count[m], reverse=True):
            if None in (mutation, mutation_types[mutation]):
                continue
            operation = getattr(mutated, f"_mutate_with_{mutation_types[mutation]}")
            mutated = operation(mutation)
        mutated.name += f" (mutations: {', '.join(mutations)})"
        mutated.metadata.update({"mutations": mutations})
        return mutated

    @staticmethod
    def _type_mutation(mutation, raise_errors=True):
        """
        Guess which kind of operation ``mutation`` is asking for.
        """
        if "ins" in mutation:
            return "insertion"
        if "del" in mutation:
            return "deletion"
        if re.search(r"([A-Z])(\d+)([A-Z])", mutation) is not None:
            return "substitution"
        if raise_errors:
            raise ValueError(f"Mutation `{mutation}` is not recognized")

    def _mutate_with_substitution(self, mutation: str) -> "Biosequence":
        """
        Given ``XYYYZ``, replace element ``X`` at position ``YYY`` with ``Z``.

        Parameters
        ----------
        mutation : str
            Replacement to apply. It must be formatted as
            ``[existing element][1-indexed position][new element]``

        Returns
        -------
        Biosequence
            Replaced sequence
        """
        # replacement: e.g. C1156Y
        search = re.search(r"([A-Z])(\d+)([A-Z])", mutation)
        if search is None:
            raise ValueError(f"Mutation `{mutation}` is not a valid substitution.")
        old, position, new = search.groups()
        assert (
            new in self.ALPHABET
        ), f"{new} is not a valid {self.__class__.__name__} character ({self.ALPHABET})"
        index = int(position) - 1
        return self.__class__(f"{self[:index]}{new}{self[index+1:]}")

    def _mutate_with_deletion(self, mutation: str) -> "Biosequence":
        """
        Given ``AXXX-BYYYdel``, delete everything between elements ``A`` and ``B`` at positions
        ``XXX`` and ``YYY``, respectively. ``A`` and ``B`` will still be part of the resulting sequence.

        Parameters
        ----------
        mutation : str
            Replacement to apply. It must be formatted as
            ``[starting element][1-indexed starting position]-[ending element][1-indexed ending position]del``

        Returns
        -------
        Biosequence
            Edited sequence
        """
        # deletion: e.g. L746-A750del
        search = re.search(r"[A-Z](\d+)-[A-Z](\d+)del", mutation)
        if search is None:
            raise ValueError(f"Mutation `{mutation}` is not a valid deletion.")
        start = int(search.group(1))
        end = int(search.group(2)) - 1
        return self.__class__(f"{self[:start]}{self[end:]}")

    def _mutate_with_insertion(self, mutation: str) -> "Biosequence":
        """
        Given ``XXXAdel``, insert element ``A`` at position ``XXX``.

        Parameters
        -----------
        mutation : str
            Insertion to apply. It must be formatted as
            ``[1-indexed insert position][element to be inserted]ins``

        Returns
        -------
        Biosequence
            Edited sequence
        """
        # insertion: e.g. 1151Tins
        search = re.search(r"(\d+)([A-Z]+)ins", mutation)
        if search is None:
            raise ValueError(f"Mutation `{mutation}` is not a valid insertion.")
        position = int(search.group(1))
        residue = search.group(2)
        assert all(r in self.ALPHABET for r in residue)
        return self.__class__(f"{self[:position]}{residue}{self[position:]}")


class DNASequence(Biosequence):
    """Biosequence that only allows DNA bases"""

    ALPHABET = "ATCG"
    _ACCESSION_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id={}&rettype=fasta&retmode=text"


class RNASequence(Biosequence):
    """Biosequence that only allows RNA bases"""

    ALPHABET = "AUCG"
    _ACCESSION_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id={}&rettype=fasta&retmode=text"


class KinaseDomainAminoAcidSequence(Biosequence):
    """Biosequence for kinase domain amino acid sequences."""

    ACCESSION_MAX_RETRIEVAL = None

    @classmethod
    def from_uniprot(
        cls,
        *uniprot_ids: str,
    ) -> Union["KinaseDomainAminoAcidSequence", Iterable["KinaseDomainAminoAcidSequence"], None]:
        """
        Retrieve kinase domain amino acid sequences of kinases defined by their Uniprot identifiers.

        Parameters
        ----------
        uniprot_ids: str
            Uniprot identifier(s). Multiple can be provided.

        Returns
        -------
        kinase_domain_sequences: list of KinaseDomainAminoAcidSequence
            Retrieved kinase domain amino acid sequence(s).
        """
        for uniprot_id in uniprot_ids:
            # request data
            response = requests.get(f"https://www.ebi.ac.uk/proteins/api/proteins/{uniprot_id}")
            protein = json.loads(response.text)

            # find protein kinase domains
            for feature in protein["features"]:
                if feature["type"] == "DOMAIN":
                    if feature["description"] == "Protein kinase":
                        # get kinase domain sequence details
                        sequence = protein["sequence"]["sequence"]
                        name = protein["id"]
                        begin = int(feature["begin"])
                        if begin == 1:
                            true_N_terminus = True
                        else:
                            true_N_terminus = False
                        end = int(feature["end"])
                        if end == len(sequence):
                            true_C_terminus = True
                        else:
                            true_C_terminus = False
                        kinase_domain_sequence = sequence[begin - 1 : end]

            yield cls(
                kinase_domain_sequence,
                name=name,
                metadata={
                    "uniprot_id": uniprot_id,
                    "begin": begin,
                    "end": end,
                    "true_N_terminus": true_N_terminus,
                    "true_C_terminus": true_C_terminus,
                },
            )

class KinasePocketAminoAcidSequence(Biosequence):
    """Biosequence for kinase pocket amino acid sequences."""

    @classmethod
    def from_uniprot(
        cls,
        *uniprot_ids: str,
    ) -> Union["KinasePocketAminoAcidSequence", Iterable["KinasePocketAminoAcidSequence"], None]:
        """
        Retrieve kinase binding site amino acid sequences of kinases defined by their Uniprot identifiers.

        Parameters
        ----------
        uniprot_ids: str
            Uniprot identifier(s). Multiple can be provided.

        Returns
        -------
        klifs_binding_site_sequences: list of KinasePocketAminoAcidSequence
            Retrieved kinase binding site amino acid sequence(s).
        """

        #def from_uniprot_to_klifs_binding_site_sequence(uniprot_ID):
        #response = requests.get(f"https://klifs.vu-compmedchem.nl/api/kinase_ID?kinase_name={uniprot_ID}&species=HUMAN")
        #if response.status_code == 200:
        #    klifs_binding_site_sequence = response.json()[0]['pocket']
        #    return klifs_binding_site_sequence
        #else:
        #    None

        for uniprot_id in uniprot_ids:
            # request data
            response = requests.get(f"https://klifs.vu-compmedchem.nl/api/kinase_ID?kinase_name={uniprot_id}&species=HUMAN")
            klifs_binding_site_sequence = response.json()[0]['pocket']

            yield cls(
                klifs_binding_site_sequence,
                metadata={
                    "uniprot_id": uniprot_id,
                },
            )
