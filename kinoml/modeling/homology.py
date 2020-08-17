from ..core.proteins import ProteinStructure
from typing import Union


class HomologyModel(): # inherent a Base class? 

    """
    Given a UniProt identifier, generate a template on to which 
    a homology model can be made and constrcut a homology model.

    The passed sequence will be used to generate a PDB template 
    structure (based on a BLAST search) or, if provided, use a 
    pre-generated template. This will be used to produce a final 
    homology model.
    """

    def __init__(self, identifier=None, template=None, sequence=None, *args, **kwargs):
        self. identifier = identifier
        self.template = template
        self.sequence = sequence


    def get_pdb_template(self, sequence):
        """
        Retrieve a template structure from PDB from a BLAST search
        Parameters
        ----------
        sequence: str
            A string of the protein sequence
        Returns
        -------
        pdb_template: ProteinStructure
            A ProteinStructure object generated from retrieval from a PDB BLAST search.
        """
        
        from prody import blastPDB
        import tempfile
        import pickle

        blast_record = blastPDB(sequence)

        with tempfile.NamedTemporaryFile(suffix='.pkl') as temp_file:
            pickle.dump(blast_record, temp_file)

            blast_record = pickle.load(open(temp_file.name, 'rb'))
            best = blast_record.getBest()['pdb_id']

            top_pdb_template = ProteinStructure.from_name(best)

        return top_pdb_template

    def get_alignment():
        raise NotImplementedError
    

    def get_model():
        raise NotImplementedError

