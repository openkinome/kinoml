"""
Test alignment functionalities of `kinoml.modeling`
"""
import pytest

from kinoml.modeling.alignment import sequence_similarity


@pytest.mark.parametrize(
    "sequence1, sequence2, similarity",
    [
        (
            "NVG",
            "NVG",
            16,
        ),
        (
            "NVG",
            "NG",
            1,
        ),
        (
            "NVG",
            "VG",
            -1,
        ),
    ],
)
def test_sequence_similarity(sequence1, sequence2, similarity):
    """Compare results to expected similarity."""
    score = sequence_similarity(sequence1, sequence2)
    assert score == similarity
