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

    def _format_alignment_with_ligands(self, template, aligned_template_seq):

        from MDAnalysis.lib.util import convert_aa_code

        sel = template.universe.select_atoms("protein and name CA")
        resnames = sel.resnames.tolist()
        resnames_one_letter = [convert_aa_code(x) for x in resnames]

        # get the last four AAs in the aligned target sequence
        end_chunk = aligned_template_seq[-4:][::-1]
        aa_store = []
        # loop over residues in reverse
        for i, letter in enumerate(resnames_one_letter[::-1]):
            # read a chunk of four AAs
            current_chunk = resnames_one_letter[::-1][i : i + 4]

            # if the end of the aligned target sequence is hit, stop
            if current_chunk == end_chunk:
                break
            # otherwise store the one letter AAs that need to be added
            else:
                aa_store.append(letter)

        # create a one element list of dashes
        dashes = "".join(["-" for x in aa_store])
        # create one element list of AAs in the canonical order
        amino_acids = "".join(aa_store[::-1])

        return amino_acids, dashes

    def make_ali_file(
        self,
        aligned_template_seq: str,
        aligned_target_seq: str,
        template: ProteinStructure,
        target: Union[str, KinaseDomainAminoAcidSequence],
        ligand: bool = False,
    ):
        """
        Generate an alignment file in MODELLER format
        ----------
        aligned_template_seq: str
            The aligned template sequence
        aligned_target_seq: str
            The aligned target sequence
        template: ProteinStructure
            The template structure
        target: list of str or KinaseDomainAminoAcidSequence
            The original target sequence
        ligand: bool
            Specify whether to retain a ligand in the alignment.
            (Default: False)
        Returns
        -------
        """

        # Convert None entries into dashes
        seq1_dashed = [i or "-" for i in aligned_template_seq]
        seq2_dashed = [i or "-" for i in aligned_target_seq]

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

        if ligand:
            # extract the AAs and No. dashes that need to be added in the alignment
            tail_amino_acids, tail_dashes = self._format_alignment_with_ligands(
                template, aligned_template_seq
            )

            # add on bufffer AAs / dashes so that Modeller finds the ligand correctly
            seq1_dashed += tail_amino_acids
            seq2_dashed += tail_dashes

        # write alignment file in MODELLER format
        with open(self.alignment_file_path, "w") as ali_file:
            for i in range(len(seq1_dashed)):
                if i == 0:
                    ali_file.write(f">P1;{protein_id}\n")
                    ali_file.write(
                        f"structure:{protein_id}:{protein_start}:A:{protein_end}: :::     :     \n"
                    )
                ali_file.write(seq1_dashed[i])
                if i == len(seq1_dashed) - 1:
                    if ligand:
                        ali_file.write("/.*")
                    else:
                        ali_file.write("*")
                if (i + 1) % max_length == 0:
                    ali_file.write("\n")

            for i in range(len(seq2_dashed)):
                # start new line below first sequence
                if i == 0:
                    ali_file.write(f"\n>P1;{sequence_id}\n")
                    if ligand:
                        ali_file.write(
                            f"sequence:{sequence_id}:{sequence_begin}: :{sequence_end + len(tail_dashes)}: :::     :     \n"
                        )
                    else:
                        ali_file.write(
                            f"sequence:{sequence_id}:{sequence_begin}: :{sequence_end}: :::     :     \n"
                        )
                ali_file.write(seq2_dashed[i])
                if i == len(seq2_dashed) - 1:
                    if ligand:
                        ali_file.write("/.*")
                    else:
                        ali_file.write("*")
                if (i + 1) % max_length == 0:
                    ali_file.write("\n")


def sequence_similarity(
        sequence1: str,
        sequence2: str,
        open_gap_penalty: int = -11,
        extend_gap_penalty: int = -1,
        substitution_matrix: str = "BLOSUM62",
) -> float:
    """
    Calculate the squence similarity of two amino acid sequences.

    Parameters
    ----------
    sequence1: str
        The first sequence.
    sequence2: str
        The second sequence.
    open_gap_penalty: int
        The penalty to open a gap.
    extend_gap_penalty: int
        The penalty to extend a gap.
    substitution_matrix: str
        The substitution matrix to use during alignment.
        Available matrices can be found via:
        >>> from Bio.Align import substitution_matrices
        >>> substitution_matrices.load()

    Returns
    -------
    score: float
        Similarity of sequences.
    """
    from Bio import pairwise2
    from Bio.Align import substitution_matrices

    substitution_matrix = substitution_matrices.load(substitution_matrix)
    # replace any characters unknown to the substitution matrix by *
    sequence1_clean = "".join(
        [x if x in substitution_matrix.alphabet else "*" for x in sequence1]
    )
    sequence2_clean = "".join(
        [x if x in substitution_matrix.alphabet else "*" for x in sequence2]
    )
    score = pairwise2.align.globalds(
        sequence1_clean,
        sequence2_clean,
        substitution_matrix,
        open_gap_penalty,
        extend_gap_penalty,
        score_only=True
    )
    return score
