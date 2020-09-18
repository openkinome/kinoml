from ..core.proteins import ProteinStructure
from ..core.sequences import KinaseDomainAminoAcidSequence
from typing import Union


class Alignment:
    """Alignment representation of protein sequences"""

    def __init__(
        self, metadata=None, alignment=None, alignment_file_path=None, *args, **kwargs
    ):

        from appdirs import user_cache_dir

        if metadata is None:
            metadata = {}
        self.metadata = metadata
        self.alignment = alignment
        self.alignment_file_path = f"{user_cache_dir()}/alignment.ali"

    @classmethod
    def get_alignment(cls, seq1: str, seq2: str, local: bool = True):
        """
        Generate an alignment between two sequences
        ----------
        seq1: str
            The first sequence to be aligned
        seq1: str
            The second sequence to be aligned
        local: bool
            If false, a global alignment is performed
            (based on the Needleman-Wunsch algorithm),
            otherwise a local alignment is performed
            (based on the Smith–Waterman algorithm).
            (Default: True)
        Returns
        -------
        Alignment
        """

        import biotite.sequence as seq
        import biotite.sequence.align as align
        import numpy as np

        # create the default matrix
        # TODO add more options for the choice of matrix
        matrix = align.SubstitutionMatrix.std_protein_matrix()

        alignments = align.align_optimal(
            seq.ProteinSequence(seq1),
            seq.ProteinSequence(seq2),
            matrix,
            local=local,
        )

        alignment = alignments[0]

        score = alignment.score
        seq_identity = align.get_sequence_identity(alignment)
        symbols = align.get_symbols(alignment)
        codes = align.get_codes(alignment)

        return cls(
            alignment=alignment,
            metadata={
                "score": score,
                "sequence_identity": seq_identity,
                "symbols": symbols,
                "codes": codes,
            },
        )

    def make_ali_file(
        self,
        aligned_seq1: str,
        aligned_seq2: str,
        template: ProteinStructure,
        target: Union[str, KinaseDomainAminoAcidSequence],
        ligand: bool = False,
    ):
        """
        Generate an alignment file in MODELLER format
        ----------
        aligned_seq1: str
            The first aligned sequence
        aligned_seq1: str
            The second aligned sequence
        template: ProteinStructure
            The template to be used in the alignment
        target: list of str or KinaseDomainAminoAcidSequence
            The target sequence to be used in the alignment
        ligand: bool
            Specify whether to retain a ligand in the alignment.
            (Default: False)
        Returns
        -------
        """

        # Convert None entries into dashes
        conv = lambda i: i or "-"
        seq1_dashed = [conv(i) for i in aligned_seq1]
        seq2_dashed = [conv(i) for i in aligned_seq2]

        # Setup formatting for MODELLER alignment file
        max_length = 75

        # TODO handle if using backend='ncbi'

        # handle if target is KinaseDomainAminoAcidSequence vs. str
        try:
            sequence_id = getattr(target, "metadata")["uniprot_id"]
            sequence_begin = getattr(target, "metadata")["begin"]
            sequence_end = getattr(target, "metadata")["end"]
        except:
            sequence_id = "sequence_id"
            sequence_begin = "1"
            sequence_end = len(target)

        protein_id = template.metadata["id"]
        # get index for the first residue in the structure sequence
        # ignoring negative and 0 resids
        resids = template.universe.residues.resids
        protein_start_index = next(
            (i for i, x in enumerate(template.universe.residues.resids) if x > 0),
            None,
        )
        protein_start = resids[protein_start_index]
        protein_end = ""

        # write alignment file in MODELLER format
        with open(f"{self.alignment_file_path}", "w") as ali_file:
            for i in range(len(seq1_dashed)):
                if i == 0:
                    ali_file.write(f">P1;{protein_id}\n")
                    ali_file.write(
                        f"structure:{protein_id}:{protein_start}:A:{protein_end}: :::     :     \n"
                    )
                ali_file.write(seq1_dashed[i])
                if i == len(seq1_dashed) - 1:
                    if ligand:
                        ali_file.write(".*")
                    else:
                        ali_file.write("*")
                if (i + 1) % max_length == 0:
                    ali_file.write("\n")

            for i in range(len(seq2_dashed)):
                # start new line below first sequence
                if i == 0:
                    ali_file.write(f"\n>P1;{sequence_id}\n")
                    ali_file.write(
                        f"sequence:{sequence_id}:{sequence_begin}: :{sequence_end}: :::     :     \n"
                    )
                ali_file.write(seq2_dashed[i])
                if i == len(seq2_dashed) - 1:
                    if ligand:
                        ali_file.write(".*")
                    else:
                        ali_file.write("*")
                if (i + 1) % max_length == 0:
                    ali_file.write("\n")