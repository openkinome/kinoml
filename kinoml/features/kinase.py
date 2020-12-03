"""
kinase_model.py
Defines the Kinase class

"""

class Kinase(object):

    def __init__(self, pdb, chain, kinase_id, name, struct_id, ligand, pocket_seq,
                 numbering, key_res, dihedrals, distances, mean_dist):
        """This script defines a Kinase class of which any kinase can be represented as an object with the
        following parameters:

        Parameters
        ----------
        pdb: str
            The PDB code of the structure.
        chain: str
            The chain index of the structure.
        kinase_id: int
            The standard ID of a kinase enforced by the KLIFS database.
        name: str
            The standard name of the kinase used by the KLIFS database.
        struct_id: int
            The ID associated with a specific chain in the pdb structure of a kinase.
        ligand: str
            The ligand name as it appears in the pdb file.
        pocket_seq: str
            The 85 discontinuous residues (from multi-sequence alignment) that define the binding pocket of a kinase.
        numbering: list of int
            The residue indices of the 85 pocket residues specific to the structure.
        key_res: list of int
            A list of residue indices that are relevant to the collective variables.
        dihedrals: list of floats
            A list (one frame) or lists (multiple frames) of dihedrals relevant to kinase conformation.
        distances: list of floats
            A list (one frame) or lists (multiple frames) of intramolecular distances relevant to kinase conformation.
        mean_dist: float
            A float (one frame) or a list of floats (multiple frames), which is the mean pairwise distance between
            ligand heavy atoms and the CAs of the 85 pocket residues.

        .. todo ::

           This is WAY too many positional arguments. Can we use kwargs instead, or somehow simplify the positional arguments into logical groups?
           Many of these will be optional if we want to represent aspects of a structure, so there's no need to make them all requiredself.
           Also, we will likely not want to mix features (distances, dihedrals) with structural information directly.

        """

        self.pdb = pdb
        self.chain = chain
        self.kinase_id = kinase_id
        self.name = name
        self.struct_id = struct_id
        self.ligand = ligand
        self.pocket_seq = pocket_seq
        self.numbering = numbering
        self.key_res = key_res
        self.dihedrals = dihedrals
        self.distances = distances
        self.mean_dist = mean_dist
