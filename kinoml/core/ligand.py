"""
Core objects to deal with ligands and small compounds
"""

from openforcefield.topology import Molecule


class Ligand(Molecule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

