"""
Core objects to deal with proteins and receptors
"""


class Protein:

    """
    Provides a chain-residue-atom model for biopolymers such
    as proteins.

    TODO: Which object should we use as an underlying representation?
    TODO: Write docstring for parameters and expected input
    """

    def __init__(self, molecule=None, name=None, sequence=None, *args, **kwargs):
        self.molecule = molecule
        self.name = name if name is not None else self._random_id()
        self.sequence = sequence

    @staticmethod
    def _random_id():
        from string import ascii_letters
        from random import choice
        return ''.join([choice(ascii_letters) for _ in range(5)])
