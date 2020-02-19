import logging

from openforcefield.topology import Molecule

logger = logging.getLogger(__name__)


class Ligand(Molecule):

    """
    TODO: Everything in this class
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
