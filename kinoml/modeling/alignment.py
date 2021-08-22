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
        score_only=True,
    )
    return score
